# 数据分析
import pandas as pd 
import numpy as np
import pickle
from tqdm import *
import statsmodels.api as sm
from scipy.stats import pearsonr
from scipy import stats

#作图包
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#中文乱码
plt.rcParams['axes.unicode_minus']=False#中文乱码
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

# 米筐
import rqsdk
from rqdatac import *
init()
import rqdatac
from rqfactor import *

# 关闭通知
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

# 研究时间
start_date = '2016-02-01'   
end_date = '2023-07-01'       
# 研究标的
index_item = '000852.XSHG'            # 【example】：全A 000985.XSHG 中证1000 000852.XSHG


""" 因子ic分析 """
def ic_analyse(ic_df):
    ic_df = ic_df.dropna(how = 'all')
    whole_ic = pd.DataFrame(ic_df.mean())
    whole_ir = pd.DataFrame(ic_df.mean()/ic_df.std())
    last_3y = ic_df.replace(0,np.nan).dropna().iloc[-250*3:]
    last_3y = pd.DataFrame(last_3y.mean()/last_3y.std())
    last_1y = ic_df.replace(0,np.nan).dropna().iloc[-250:]
    last_1y = pd.DataFrame(last_1y.mean()/last_1y.std())
    last_3m = ic_df.replace(0,np.nan).dropna().iloc[-60:]
    last_3m = pd.DataFrame(last_3m.mean()/last_3m.std())
    compare = pd.concat([whole_ic,whole_ir,last_3y,last_1y,last_3m],axis = 1)
    compare.columns = ['IC','IR','last_3y','last_1y','last_3m']

    return compare

# 累计ic图
def cumic(name,ic_df):
    """
    :param name: 因子名称 -> list 
    :param ic_df: ic序列表 -> dataframe 
    :return fig: 累计ic图 -> plot
    """
    ic_df[name].cumsum().plot()

# 热力图    
def hot_corr(name,ic_df):
    """
    :param name: 因子名称 -> list 
    :param ic_df: ic序列表 -> dataframe 
    :return fig: 热力图 -> plt
    """
    ax = plt.subplots(figsize=(len(name), len(name)))#调整画布大小
    ax = sns.heatmap(ic_df[name].corr(),vmin=0.4, square=True, annot= True,cmap = 'Blues')   #annot=True 表示显示系数
    plt.title('Factors_IC_CORRELATION')
    # 设置刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

# 数据清洗函数 -----------------------------------------------------------
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
        df_industy_market = pd.read_pickle(f'df_industy_market_{start}_{end}.pkl')
    except:
        market_cap = execute_factor(LOG(Factor('market_cap_3')),order_book_ids,start,end).stack().to_frame('market_cap')
        industry_df = get_industry_exposure(order_book_ids,datetime_period)
        #合并因子
        df_industy_market = pd.concat([market_cap,industry_df],axis = 1)
        df_industy_market.index.names = ['datetime','order_book_id']
        df_industy_market.dropna(inplace = True)
        df_industy_market.to_pickle(f'df_industy_market_{start}_{end}.pkl')

    df_industy_market = pd.concat([df.stack().to_frame('factor'),df_industy_market],axis = 1).dropna()
    #OLS回归
    df_result = pd.DataFrame()
    for i in tqdm(datetime_period):
        df_day = df_industy_market.loc[i]
        x = df_day.iloc[:,1:]   #市值/行业
        y = df_day.iloc[:,0]    #因子值
        df_day_result = pd.DataFrame(sm.OLS(y.astype(float),x.astype(float),hasconst=False, missing='drop').fit().resid,columns=[i])
        df_result = pd.concat([df_result,df_day_result],axis = 1)
    df_result = df_result.T
    df_result.index.names = ['datetime']

    return df_result


def get_industry_exposure(order_book_ids,datetime_period):
    
    """
    :param order_book_ids: 股票池 -> list
    :param datetime_period: 研究日 -> list
    :return result: 虚拟变量 -> dataframe
    """
    
    zx2019_industry = rqdatac.client.get_client().execute('__internal__zx2019_industry')
    df = pd.DataFrame(zx2019_industry)
    df.set_index(['order_book_id', 'start_date'], inplace=True)
    df = df['first_industry_name'].sort_index()
    print('中信行业数据已获取')

    #构建动态行业数据表格
    index = pd.MultiIndex.from_product([order_book_ids, datetime_period], names=['order_book_id', 'datetime'])
    pos = df.index.searchsorted(index, side='right') - 1
    index = index.swaplevel()   # level change (oid, datetime) --> (datetime, oid)
    result = pd.Series(df.values[pos], index=index)
    result = result.sort_index()
    print('动态行业数据已构建')
    
    #生成行业虚拟变量
    return pd.get_dummies(result)


def data_clean(df):
    return standardize(neutralization(standardize(mad(df))))

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
        close = pd.read_pickle(f'close_{start}_{end}.pkl')
    except:
        close = get_price(order_book_ids, start_date=start, end_date=end,frequency='1d',fields='close').close.unstack('order_book_id')
        close.to_pickle(f'close_{start}_{end}.pkl')
    
    return_n = close.pct_change(n).shift(-n)

    if Rank_IC == True:
        x = df.corrwith(return_n,axis = 1,method='spearman').dropna(how = 'all')
    else:
        x = df.corrwith(return_n,axis = 1,method='pearson').dropna(how = 'all')
    
    t_stat, p_value = stats.ttest_1samp(x, 0)
    
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


#### 分层效应
def top_ret(df,name = '',n = 20,g = 10):

    """
    :param df: 因子值 -> unstack
    :param df: 因子名称 -> str
    :param n: 调仓日 -> int
    :param g: 分组数量 -> int 
    :return group_return: 各分组日收益率 -> dataframe
    """
    
    order_book_ids = df.columns.tolist()
    datetime_period = df.index.tolist()
    start = datetime_period[0].strftime('%Y-%m-%d')
    end = datetime_period[-1].strftime('%Y-%m-%d')
    try:
        return_1d = pd.read_pickle(f'return_1d_{start}_{end}.pkl')
    except:
        return_1d = get_price(order_book_ids, start_date=start, end_date=end,frequency='1d',fields='close').close.unstack('order_book_id').pct_change().shift(-1).dropna(axis = 0,how = 'all').stack()
        return_1d.to_pickle(f'return_1d_{start}_{end}.pkl')

    group = df.stack().to_frame('factor')
    group['return_1d'] = return_1d
    group.dropna(inplace = True)
    group.reset_index(inplace = True)
    group.columns = ['date','stock','factor','current_renturn']

    year_ret_df = pd.DataFrame()
    for i in tqdm(range(0,len(datetime_period)-1,n)):
        # 调仓
        single = group[group.date == datetime_period[i]].sort_values(by = 'factor')
        
        # 分组
        single.loc[:,'group'] = pd.qcut(single.factor,g, list(range(1,g+1))).to_list()  # N 组数
        
        # 获取周期
        if i <= len(datetime_period)-n:
            period = group[group.date.isin(datetime_period[i:i+n])]
        else:
            period = group[group.date.isin(datetime_period[i:])]

        # 计算各分组收益率
        year_ret_df_temp = period[period.stock.isin(single[single.group == g].stock.tolist())].set_index(['date','stock']).current_renturn.unstack('stock').mean(axis = 1)

        year_ret_df = pd.concat([year_ret_df,year_ret_df_temp],axis = 0)
    year_ret = ((year_ret_df + 1).cumprod().iloc[-1] ** (252/year_ret_df.shape[0]))
        
    return year_ret[0] - 1

# 含手续费
def BACKTEST(df,name = '',n = 50,change_days = 5,tax = 0.001,commission = 0.0002,benchmark = '000906.XSHG',fix = False,limit_n = 1,filter = False):

    """
    :param df: 因子值 -> unstack
    :param name: 因子名称 -> str
    :param n: 最多选取因子值最大的n只标的 -> int 
    :param change_days: 调仓周期 -> int
    :param tax: 印花税（默认千一） -> float
    :param commission: 交易佣金（默认万二） -> int
    :param benchmark: 基准 -> str
    :param fix: 是否修正买入队列 -> bool
    :param limit_n: 数据修正填充数量 -> int
    :param filter: 是否过滤ST涨停停牌 -> bool
    :return net: 净值序列 -> dataframe
    :return performance: 绩效分析 -> dataframe
    """

    """ 修正买入队列 """
    if fix == True:
        def fix_cycle(df):
            #匹配研究周期交易日数据
            start_date = df.index[0]
            end_date = df.index[-1]
            trade_days = pd.to_datetime(get_trading_dates(start_date, end_date))
            trade_days = pd.DataFrame([None]*len(trade_days),index = trade_days,columns = ['fix_day'])
            #合并修改原表格，并向后填充，删除匹配None队列
            df = pd.concat([df,trade_days],axis = 1).fillna(method = 'ffill',limit = limit_n).dropna(axis = 1,how ='all')
            #截取研究周期
            df = df.loc[trade_days.index.tolist()]
            df.dropna(how ='all',inplace = True)

            return df
    
        df = fix_cycle(df)
    else:
        pass
    
    # 获取时间标的
    stock_list = df.columns.tolist()
    date_list = df.index.tolist()

    """ 数据ST涨停停牌过滤 """
    def get_st_filter(stock_list,date_list):
        # 对st股票做标记,st=1,非st=0

        st_filter = is_st_stock(stock_list,date_list[0],date_list[-1]).astype('float').reindex(columns=stock_list,index = date_list)                                #剔除ST
        st_filter.replace(1,True,inplace = True)
        st_filter.replace(0,False,inplace = True)
        st_filter = st_filter.shift(-1).fillna(method = 'ffill')
        print('剔除ST已构建')

        return st_filter

    def get_suspended_filter(stock_list,date_list):

        suspended_filter = is_suspended(stock_list,date_list[0],date_list[-1]).astype('float').reindex(columns=stock_list,index=date_list)

        suspended_filter.replace(1,True,inplace = True)
        suspended_filter.replace(0,False,inplace = True)
        suspended_filter = suspended_filter.shift(-1).fillna(method = 'ffill')
        print('剔除停牌已构建')

        return suspended_filter

    def get_limit_up_down_filter(stock_list,date_list):

        # 涨停则赋值为1,反之为0    
        df = pd.DataFrame(index = date_list,columns=stock_list,data=0.0)
        total_price = get_price(stock_list,date_list[0],date_list[-1],adjust_type='none')

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
    
    if filter == True:
        st_filter = get_st_filter(stock_list,date_list)
        suspended_filter = get_suspended_filter(stock_list,date_list)
        limit_up_down_filter = get_limit_up_down_filter(stock_list,date_list)

        df = df.mask(suspended_filter).mask(st_filter).mask(limit_up_down_filter)
    else:
        pass

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
    print('\n correlation: {}'.format(net.corr().iloc[0,1]))
    cum_net = (net+1).cumprod()
    cum_net['alpha'] = cum_net['net'] / cum_net['benchmark']
    cum_net.dropna(inplace = True)
    cum_net.plot(figsize = (10,6),title = f'back_test_{name}',secondary_y = 'alpha')

    """ 策略回测报告 """

    def get_Performance_analysis(T,year_day = 252,turnover_rate = turnover_rate):
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

        turnover_rate = sum(turnover_rate)/len(timelist)
        
        # 输出
        return [net_values,year_ret_sqrt,Sharpe,downlow,volitiy,turnover_rate]

    performance = pd.DataFrame()
    for i in cum_net.columns.tolist():
        temp = pd.DataFrame(get_Performance_analysis(cum_net[i]),columns= [i]).T
        performance = pd.concat([performance,temp],axis = 0)
    performance.columns = ['净值','年化收益率','夏普比率','最大回测','波动率','日平均换手率']
    
    return net,performance