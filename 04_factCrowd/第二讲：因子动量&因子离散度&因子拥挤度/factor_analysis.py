"""
update:2023/09/24
author:Nick_Ni
"""

import pandas as pd 
import numpy as np
from scipy import stats
import statsmodels.api as sm
from tqdm import *

from rqdatac import *
from rqfactor import *
from rqfactor.notebook import *
from rqfactor.extension import *
from rqfactor import Factor,LOG,REF,STD
init()
import rqdatac

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 动态券池
def INDEX_FIX(start_date,end_date,index_item):
    """
    :param start_date: 开始日 -> str
    :param end_date: 结束日 -> str 
    :param index_item: 指数代码 -> str 
    :return index_fix: 动态因子值 -> unstack
    """
    
    index = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in index_components(index_item,start_date= start_date,end_date=end_date).items()])).T

    # 构建动态股票池 
    index_fix = index.unstack().reset_index().iloc[:,-2:]
    index_fix.columns = ['date','stock']
    index_fix.date = pd.to_datetime(index_fix.date)
    index_fix['level'] = True
    index_fix.dropna(inplace = True)
    index_fix = index_fix.set_index(['date','stock']).level.unstack()
    index_fix.fillna(False,inplace = True)

    return index_fix

# 再次定义函数：计算最大回撤
def get_Performance_analysis(T,year_day = 252):
    # 获取最终净值
    net_values = round(T[-1],4)
    
    # 计算几何年化收益率
    year_ret_sqrt = net_values**(year_day/len(T))-1
    year_ret_sqrt = round(year_ret_sqrt*100,2)
    
    # 计算年化波动率
    volitiy = T.pct_change().dropna().std()*np.sqrt(year_day)
    volitiy = round(volitiy*100,2)
    
    #计算夏普，无风险收益率记3%
    Sharpe = (year_ret_sqrt - 3)/volitiy
    Sharpe = round(Sharpe,2)

    # 计算最大回撤
    # 最大回撤结束点
    i = np.argmax((np.maximum.accumulate(T) - T)/np.maximum.accumulate(T))
    # 开始点
    j = np.argmax(T[:i])

    downlow = round((1-T[i]/T[j])*100,2)

    # 输出
    return [net_values,year_ret_sqrt,Sharpe,downlow,volitiy]
#------------------------------------------------------------------------

# 券池过滤
def get_new_stock_filter(stock_list,date_list, newly_listed_threshold = 252):

    listed_date_list = [rqdatac.instruments(stock).listed_date for stock in stock_list]        
    newly_listed_window = pd.Series(index=stock_list, data=[rqdatac.get_next_trading_date(listed_date, n=newly_listed_threshold) for listed_date in listed_date_list])     
    newly_listed_label = pd.DataFrame(index=date_list, columns=stock_list, data=0.0)

    # 上市时间短语指定窗口的新股标记为1，否则为0
    for stock in newly_listed_window.index:
        newly_listed_label.loc[:newly_listed_window.loc[stock], stock] = 1.0
                    #剔除新股
    newly_listed_label.replace(1,True,inplace = True)
    newly_listed_label.replace(0,False,inplace = True)
    newly_listed_label = newly_listed_label.shift(-1).fillna(method = 'ffill')
    print('剔除新股已构建')

    return newly_listed_label

def get_st_filter(stock_list,date_list):
    # 对st股票做标记,st=1,非st=0

    st_filter = rqdatac.is_st_stock(stock_list,date_list[0],date_list[-1]).astype('float').reindex(columns=stock_list,index = date_list)                                #剔除ST
    st_filter.replace(1,True,inplace = True)
    st_filter.replace(0,False,inplace = True)
    st_filter = st_filter.shift(-1).fillna(method = 'ffill')
    print('剔除ST已构建')

    return st_filter

def get_suspended_filter(stock_list,date_list):

    suspended_filter = rqdatac.is_suspended(stock_list,date_list[0],date_list[-1]).astype('float').reindex(columns=stock_list,index=date_list)

    suspended_filter.replace(1,True,inplace = True)
    suspended_filter.replace(0,False,inplace = True)
    suspended_filter = suspended_filter.shift(-1).fillna(method = 'ffill')
    print('剔除停牌已构建')

    return suspended_filter

def get_limit_up_down_filter(stock_list,date_list):

    # 涨停则赋值为1,反之为0    
    df = pd.DataFrame(index = date_list,columns=stock_list,data=0.0)
    total_price = rqdatac.get_price(stock_list,date_list[0],date_list[-1],adjust_type='none')

    for stock in stock_list:

        try:
            price = total_price.loc[stock]
        except:
            print('no stock data:',stock)
            df[stock] = np.nan
            continue                    

        # 如果close == limit_up or limit down,则股票涨停或者跌停        
        condition = ((price['open'] == price['limit_up']))#|(price['close'] == price['limit_down']))        
        if condition.sum()!=0:
            df.loc[condition.loc[condition==True].index,stock] = 1.0

    df.replace(1,True,inplace = True)
    df.replace(0,False,inplace = True)
    df = df.shift(-1).fillna(method = 'ffill')
    print('剔除开盘涨停已构建')

    return df

# 数据清洗函数 -----------------------------------------------------------
# MAD:中位数去极值
def mad(df):
    # MAD:中位数去极值
    def filter_extreme_MAD(series,n): 
        median = series.median()
        new_median = ((series - median).abs()).median()
        return series.clip(median - n*new_median,median + n*new_median)
    # 离群值处理
    df = df.apply(lambda x :filter_extreme_MAD(x,3), axis=1)

    return df

def standardize(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

import os

def create_dir_not_exist(path):
    # 若不存在该路径则自动生成
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


def neutralization(df):

    """
    :param df: 因子值 -> unstack
    :param df_result: 中性化后的因子值 -> unstack
    """

    order_book_ids = df.columns.tolist()
    datetime_period = df.index.tolist()
    start = datetime_period[0].strftime("%Y-%m-%d")
    end = datetime_period[-1].strftime("%Y-%m-%d")
    #获取行业/市值暴露度
    try:
        df_industy_market = pd.read_pickle(f'tmp/df_industy_market_{start}_{end}.pkl')
    except:
        market_cap = execute_factor(LOG(Factor('market_cap_3')),order_book_ids,start,end).stack().to_frame('market_cap')
        industry_df = get_industry_exposure(order_book_ids,datetime_period)
        #合并因子
        industry_df['market_cap'] = market_cap
        df_industy_market = industry_df
        df_industy_market.index.names = ['datetime','order_book_id']
        df_industy_market.dropna(inplace = True)
        create_dir_not_exist('tmp')
        df_industy_market.to_pickle(f'tmp/df_industy_market_{start}_{end}.pkl')

    df_industy_market['factor'] = df.stack()
    df_industy_market.dropna(subset = 'factor',inplace = True)
    
    #OLS回归
    df_result = pd.DataFrame(columns = order_book_ids,index = datetime_period)
    for i in tqdm(datetime_period):
        try:
            df_day = df_industy_market.loc[i]
            x = df_day.iloc[:,:-1]   #市值/行业
            y = df_day.iloc[:,-1]    #因子值
            df_result.loc[i] = sm.OLS(y.astype(float),x.astype(float),hasconst=False, missing='drop').fit().resid
        except:
            pass
    df_result.index.names = ['datetime']

    return df_result


def get_industry_exposure(order_book_ids,datetime_period):
    
    """
    :param order_book_ids: 股票池 -> list
    :param datetime_period: 研究日 -> list
    :return result: 虚拟变量 -> dataframe
    """
    print('gen industry martix... ')
    zx2019_industry = rqdatac.client.get_client().execute('__internal__zx2019_industry')
    df = pd.DataFrame(zx2019_industry)
    df.set_index(['order_book_id', 'start_date'], inplace=True)
    df = df['first_industry_name'].sort_index()
    
    #构建动态行业数据表格
    index = pd.MultiIndex.from_product([order_book_ids, datetime_period], names=['order_book_id', 'datetime'])
    pos = df.index.searchsorted(index, side='right') - 1
    index = index.swaplevel()   # level change (oid, datetime) --> (datetime, oid)
    result = pd.Series(df.values[pos], index=index)
    result = result.sort_index()
    
    #生成行业虚拟变量
    return pd.get_dummies(result)

def data_clean(df):
    df = standardize(neutralization(standardize(mad(df))))
    df = df.apply(lambda x: x.astype(float))
    return df

# 单因子检测函数 -----------------------------------------------------------

# IC计算  
def Quick_Factor_Return_N_IC(df,n,name = '',Rank_IC = True):

    """
    :param df: 因子值 -> unstack
    :param n: 调仓日 -> int
    :param True/False: Rank_ic/Normal_ic -> bool
    :return ic: IC序列 -> dataframe
    """

    order_book_ids = df.columns.tolist()
    datetime_period = df.index.tolist()
    start = datetime_period[0].strftime('%Y-%m-%d')
    end = datetime_period[-1].strftime('%Y-%m-%d')
    try:
        close = pd.read_pickle(f'tmp/close_{start}_{end}.pkl')
    except:
        close = get_price(order_book_ids, start_date=start, end_date=end,frequency='1d',fields='close').close.unstack('order_book_id')
        create_dir_not_exist('tmp')
        close.to_pickle(f'tmp/close_{start}_{end}.pkl')
    
    return_n = close.pct_change(n).shift(-n)

    if Rank_IC == True:
        x = df.corrwith(return_n,axis = 1,method='spearman').dropna(how = 'all')
    else:
        x = df.corrwith(return_n,axis = 1,method='pearson').dropna(how = 'all')
    
    t_stat,_ = stats.ttest_1samp(x, 0)
    
    IC = {'name': name,
    'IC mean':round(x.mean(),4),
    'IC std':round(x.std(),4),
    'IR':round(x.mean()/x.std(),4),
    'IR_ly':round(x[-252:].mean()/x[-252:].std(),4),
    'IC>0':round(len(x[x>0].dropna())/len(x),4),
    'ABS_IC>2%':round(len(x[abs(x) > 0.02].dropna())/len(x),4),
    't_stat':round(t_stat,4),
    }
    
    print(IC)
    IC = pd.DataFrame([IC])
    
    return x,IC


# 分层效应

def group_g(df,n,g):

    """
    :param df: 因子值 -> unstack
    :param n: 调仓日 -> int
    :param g: 分组数量 -> int 
    :return group_return: 各分组日收益率 -> dataframe
    :return turnover_ratio: 各分组日调仓日换手率 -> dataframe
    """

    order_book_ids = df.columns.tolist()
    datetime_period = df.index.tolist()
    start = datetime_period[0].strftime('%Y-%m-%d')
    end = datetime_period[-1].strftime('%Y-%m-%d')
    try:
        return_1d = pd.read_pickle(f'tmp/return_1d_{start}_{end}.pkl')
    except:
        return_1d = get_price(order_book_ids,get_previous_trading_date(start,1,market='cn'),end,
                                '1d','close','pre',False,True).close.unstack('order_book_id').pct_change().shift(-1).dropna(axis = 0,how = 'all').stack()
        create_dir_not_exist('tmp')
        return_1d.to_pickle(f'tmp/return_1d_{start}_{end}.pkl')

    group = df.stack().to_frame('factor')
    group['current_renturn'] = return_1d
    group = group.dropna()
    group.reset_index(inplace = True)
    group.columns = ['date','stock','factor','current_renturn']

    turnover_ratio = pd.DataFrame()
    group_return = pd.DataFrame()

    for i in range(0,len(datetime_period),n):
        # 调仓
        single = group[group.date == datetime_period[i]].sort_values(by = 'factor')
        
        # 分组
        single.loc[:,'group'] = pd.qcut(single.factor,g, list(range(1,g+1))).to_list()  # N 组数
        
        # 计算分组标的
        group_dict = {}
        for j in range(1,g+1):
            group_dict[j] = single[single.group == j].stock.tolist()
        
        # 计算换手率
        turnover_ratio_temp = []
        if i == 0:
            temp_group_dict = group_dict
        else:
            for j in range(1,g+1):
                turnover_ratio_temp.append(len(list(set(temp_group_dict[j]).difference(set(group_dict[j]))))/len(set(temp_group_dict[j])))
            turnover_ratio = pd.concat([turnover_ratio,pd.DataFrame(turnover_ratio_temp,index = ['G{}'.format(j) for j in list(range(1,g+1))],columns = [datetime_period[i]]).T],axis = 0)
            temp_group_dict = group_dict
        
        # 获取周期
        if i < len(datetime_period)-n:
            period = group[group.date.isin(datetime_period[i:i+n])]
        else:
            period = group[group.date.isin(datetime_period[i:])]

        # 计算各分组收益率
        group_return_temp = []
        for j in range(1,g+1):
            group_return_temp.append(period[period.stock.isin(group_dict[j])].set_index(['date','stock']).current_renturn.unstack('stock').mean(axis = 1))
        group_return = pd.concat([group_return,pd.DataFrame(group_return_temp,index = ['G{}'.format(j) for j in list(range(1,g+1))]).T],axis = 0)
        print('\r 当前：{} / 总量：{}'.format(i,len(datetime_period)),end='')
    
    return group_return,turnover_ratio

# 含手续费
def BACKTEST(df,name = '',n = 100,change_days = 20,tax = 0.0005,commission = 0.0002,benchmark = '000300.XSHG',fig = False):  
    """ 生成买入队列 """
    timelist = sorted(df.index.tolist())
    buy_stock = pd.DataFrame()
    for i in range(0,len(timelist)-2,change_days):
        #前一日存在买入信号，翌日停牌，替补
        buy_in_list = df.iloc[i:i+2,].T.dropna().sort_values(by = timelist[i],ascending = False).iloc[:n,0:1]
        buy_in_list.columns = [timelist[i+1]]
        buy_stock = pd.concat([buy_stock,buy_in_list],axis = 1)
    buy_stock[buy_stock >= -np.inf] = 1
    buy_stock = buy_stock/buy_stock.count()

    """ 计算收益（含手续费）"""
    change_day = buy_stock.columns.tolist() + [df.index[-1]]
    net = pd.DataFrame()
    net_help = pd.DataFrame()
    num = len(change_day)
    turnover_rate = []
    for i in tqdm(range(len(change_day)-1)):
        start_date = change_day[i]
        end_date = change_day[i+1]

        # 手续费计算
        current_weight = buy_stock.iloc[:,i]
        if i == 0:
            orignial_weight = pd.Series([0] * len(current_weight),index = current_weight.index)
        else:
            orignial_weight = buy_stock.iloc[:,i-1]
        weight_change = pd.DataFrame((current_weight.fillna(0) - orignial_weight.fillna(0)).replace(0,np.nan).dropna().sort_values())

        # 换手率
        if i == 0:
            turnover_rate.append(1)
        else:
            turnover_rate.append(weight_change.iloc[:,0].abs().sum())

        weight_change.columns = ['weight_change']
        cost_ratio = [(tax + commission)  for i in weight_change.weight_change.tolist() if i < 0]
        weight_change['cost_ratio'] = cost_ratio + [commission]*(len(weight_change)-len(cost_ratio))
        weight_change['cost'] = weight_change['weight_change'].abs() * weight_change['cost_ratio']
        cost = weight_change.cost.sum()
        # 防止当日无持仓
        try:
            # 收益计算
            net_temp = get_price(buy_stock.iloc[:,i].dropna().index.tolist(),start_date,end_date,
                            frequency='1d', fields='open', adjust_type='pre', skip_suspended =False, 
                            market='cn', expect_df=True).unstack().T.pct_change().shift(-1).dropna().mean(axis =1).droplevel(0).iloc[:-1]
        except:
            net_temp = pd.DataFrame([0]*change_days,index = pd.to_datetime(get_trading_dates(start_date, end_date))[:-1])
        # 扣减手续费成本
        net_temp.iloc[0] -= cost
        if i < int(num/2):
            net = pd.concat([net,net_temp],axis = 0)
        else:
            net_help = pd.concat([net_help,net_temp],axis = 0)
    net = pd.concat([net,net_help],axis = 0)
    net.columns = ['net']

    """ 获取基准"""
    net['benchmark'] = get_price([benchmark],start_date=change_day[0],end_date=df.index[-1],fields = ['open'],expect_df=True).open.pct_change().shift(-1).dropna().droplevel(0)
    #print('\n correlation: {}'.format(net.corr().iloc[0,1]))
    cum_net = (net+1).cumprod()
    cum_net['alpha'] = cum_net['net'] / cum_net['benchmark']
    cum_net.dropna(inplace = True)
    if fig == True:
        cum_net.plot(figsize = (10,6),title = f'back_test_{name}',secondary_y = 'alpha')
    else:
        pass

    """ 策略回测报告 """

    def get_Performance_analysis(T,benchmark,year_day = 252,turnover_rate = turnover_rate):
        # 获取最终净值
        net_values = round(T[-1],4)

        # 计算最大回撤
        # 最大回撤结束点
        i = np.argmax((np.maximum.accumulate(T) - T)/np.maximum.accumulate(T))
        # 开始点
        j = np.argmax(T[:i])

        downlow = round((1-T[i]/T[j])*100,2)
        
        # 计算几何年化收益率
        year_ret_sqrt = net_values**(year_day/len(T))-1
        year_ret_sqrt = round(year_ret_sqrt*100,2)
        
        # 计算年化波动率
        volitiy = T.pct_change().dropna().std()*np.sqrt(year_day)
        volitiy = round(volitiy*100,2)

        # excess_return
        excess_return = T.pct_change() - benchmark.pct_change()

        excess_year_ret_sqrt = ((excess_return + 1).cumprod()).iloc[-1]**(year_day/len(T))-1
        excess_year_ret_sqrt = round(excess_year_ret_sqrt*100,2)

        # tracking error
        tr = (excess_return).std()*np.sqrt(year_day)
        
        # 计算夏普，无风险收益率记3%
        Sharpe = (year_ret_sqrt - 3)/volitiy
        Sharpe = round(Sharpe,2)

        # 信息比率
        ir = (year_ret_sqrt - 3)/tr/100
        ir = round(ir,2)

        # calmar
        calmar = (year_ret_sqrt - 3)/downlow
        calmar = round(calmar,2)

        

        turnover_rate = sum(turnover_rate)/len(timelist)
        
        # 输出
        return [net_values,year_ret_sqrt,excess_year_ret_sqrt,Sharpe,tr,ir,downlow,calmar,turnover_rate]

    performance = pd.DataFrame()
    for i in ['net']:
        temp = pd.DataFrame(get_Performance_analysis(cum_net[i],cum_net['benchmark']),columns= [i]).T
        performance = pd.concat([performance,temp],axis = 0)
    performance.columns = ['净值','年化收益率','超额年化收益率','夏普比率','跟踪误差','信息比率','最大回测','卡玛比率','日平均换手率']
    
    return net,performance

def get_Performance_analysis(T,year_day = 252):
    # 获取最终净值
    net_values = round(T[-1],4)
    
    # 计算几何年化收益率
    year_ret_sqrt = net_values**(year_day/len(T))-1
    year_ret_sqrt = round(year_ret_sqrt*100,2)
    
    # 计算年化波动率
    volitiy = T.pct_change().dropna().std()*np.sqrt(year_day)
    volitiy = round(volitiy*100,2)
    
    #计算夏普，无风险收益率记3%
    Sharpe = (year_ret_sqrt - 3)/volitiy
    Sharpe = round(Sharpe,2)

    # 计算最大回撤
    # 最大回撤结束点
    i = np.argmax((np.maximum.accumulate(T) - T)/np.maximum.accumulate(T))
    # 开始点
    j = np.argmax(T[:i])

    downlow = round((1-T[i]/T[j])*100,2)

    # 输出
    return [net_values,year_ret_sqrt,Sharpe,downlow,volitiy]