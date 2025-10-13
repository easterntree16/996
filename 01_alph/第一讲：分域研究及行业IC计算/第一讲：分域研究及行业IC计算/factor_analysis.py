import pandas as pd 
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import statsmodels.api as sm
from tqdm import *

from rqdatac import *
from rqfactor import *
from rqfactor.notebook import *
from rqfactor.extension import *
init()
import rqdatac


def index_fix(start_date,end_date,index_item):
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

def nonlinear_transform(factor, quantile_to_subtract):
    """
    :param factor: 原始因子值 -> unstack
    :param quantile_to_subtract: 分位数 转换后因子值 = -（因子值 - 对应分位数）^2，此时因子值离该分位数越近，转换后因子值越大 -> float 
    :return new_factor: 改造后因子值 -> unstack
    """
    return -(factor.sub(factor.quantile(quantile_to_subtract, axis=1), axis=0) ** 2)

# 设置函数：计算净值曲线的绩效指标
def get_Performance_analysis(T,year_day = 252):
    """
    :param T: 净值序列 和基准净值序列 -> DataFrame
    :return net_values,year_ret_sqrt,Sharpe,downlow,volitiy: 绩效指标 -> DataFrame
    """
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

def get_new_stock_filter(stock_list,date_list, newly_listed_threshold=240):

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
    print('new_stock_filter load')

    return newly_listed_label

def get_st_filter(stock_list,date_list):
    # 对st股票做标记,st=1,非st=0

    st_filter = rqdatac.is_st_stock(stock_list,date_list[0],date_list[-1]).astype('float').reindex(columns=stock_list,index = date_list)                                #剔除ST
    st_filter.replace(1,True,inplace = True)
    st_filter.replace(0,False,inplace = True)
    st_filter = st_filter.shift(-1).fillna(method = 'ffill')
    print('st_filter load')

    return st_filter

def get_suspended_filter(stock_list,date_list):

    suspended_filter = rqdatac.is_suspended(stock_list,date_list[0],date_list[-1]).astype('float').reindex(columns=stock_list,index=date_list)

    suspended_filter.replace(1,True,inplace = True)
    suspended_filter.replace(0,False,inplace = True)
    suspended_filter = suspended_filter.shift(-1).fillna(method = 'ffill')
    print('suspended_filter load')

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
    print('limit_up_down_filter load')

    return df

# 数据清洗函数 -----------------------------------------------------------
# MAD:中位数去极值
def filter_extreme_MAD(series,n): 
    median = series.median()
    new_median = ((series - median).abs()).median()
    return series.clip(median - n*new_median,median + n*new_median)

def winsorize_std(series, n=3):
    mean, std = series.mean(), series.std()
    return series.clip(mean - std*n, mean + std*n)


def winsorize_percentile(series, left=0.025, right=0.975):
    lv, rv = np.percentile(series, [left*100, right*100])
    return series.clip(lv, rv)

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



# IC计算 / 包括行业计算
def Factor_Return_N_IC(df,n,name = '',Rank_IC = True,industry_group = True):

    """
    :param df: 因子值 -> unstack
    :param n: 调仓日 -> int
    :param Rank_IC: TrueRank_ic FalseNormal_ic -> bool
    :param industry_group: True行业聚类 False所有行业 -> bool
    :return x: IC序列 -> dataframe
    :return half_life:半衰期 -> dataframe
    :return industry_ic_analyse: 中信行业ic检验值 -> dataframe
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
    
    # 半衰期
    half_life = {}
    print('loading half_life ... ')
    for i in tqdm(range(1,21)):
        half_life[i] = df.corrwith(close.pct_change().shift(-i),axis = 1,method='spearman').dropna(how = 'all').mean()

    half_life = pd.DataFrame(half_life.items())
    half_life.columns = ['day','ic']
    half_life.set_index(['day'],inplace = True)

    plt.figure(figsize=(18,10))
    plt.subplot(2, 2, 1)
    plt.bar(half_life.index,half_life.ic)
    plt.title(f'{name}_half_life')

    # 逐季度ic
    plt.subplot(2, 2, 2)
    month_ic = (x.resample('M').mean()/x.resample('M').std()).to_frame('m_ic')
    plt.bar(month_ic.index,month_ic.m_ic,width = 30)
    plt.title(f'{name}_ir_3m')

    t_stat, p_value = stats.ttest_1samp(x, 0)
    
    print(['IC mean:{}'.format(round(x.mean(),4)),
            'IC std:{}'.format(round(x.std(),4)),
            'IR:{}'.format(round(x.mean()/x.std(),4)),
            'IR_LAST_1Y:{}'.format(round(x[-240:].mean()/x[-240:].std(),4)),
            'IC>0:{}'.format(round(len(x[x>0].dropna())/len(x),4)),
            'ABS_IC>2%:{}'.format(round(len(x[abs(x) > 0.02].dropna())/len(x),4)),
            't_stat:{}'.format(t_stat.round(4))
           ]) 
    
    # 累计ic
    plt.subplot(2, 2, 4)
    plt.plot(x.cumsum())
    plt.title(f'{name}_cumic')

    industry_group_name = {'finance':['银行','非银行金融','综合金融'],
                        'special_consumption' : ['汽车','家电','食品饮料','餐饮旅游'],
                        'normal_consumption' : ['农林牧渔','纺织服装','医药','商贸零售','消费者服务','轻工制造'],
                        'technology' : ['电子','通信','计算机','传媒'],
                        'up_cycle' : ['石油石化','煤炭','有色金属'],
                        'mid-cycle' : ['基础化工','钢铁','建材','机械','电力设备及新能源','国防军工','电力设备', '电子元器件'],
                        'down-cycle' : ['建筑','房地产','电力及公用事业','交通运输'],
                        }
    try:
        industry = pd.read_pickle(f'industry_{start}_{end}.pkl')
    except:
        industry = get_industry_exposure(order_book_ids,datetime_period).unstack()
        industry.to_pickle(f'industry_{start}_{end}.pkl')

    if industry_group == True:
        # 聚类分组后行业
        industry_ic_analyse = {}
        for i in list(industry_group_name.keys()):
            for j in industry_group_name[i]:
                industry.replace(j,i,inplace = True)
        for i in tqdm(list(industry_group_name.keys())):
            industry_group_ic = close.mask(industry[industry == i].isnull()).corrwith(df,axis = 1,method='spearman').dropna(how = 'all')
            industry_ic_analyse[i] = [industry_group_ic.mean(),industry_group_ic.mean()/industry_group_ic.std()]

        industry_ic_analyse = pd.DataFrame(industry_ic_analyse).T
        industry_ic_analyse.columns = ['IC','IR']
        # 行业IC/IR
        plt.subplot(2, 2, 3)
        plt.bar(industry_ic_analyse.index,industry_ic_analyse.IC)
        plt.xticks(rotation=90)
        plt.title(f'{name}_citics_industry_group_ic')

    else:
        # 各行业表现
        industry_name = list(set(industry.stack()))
        industry_ic_analyse = {}
        print('loading industry ... ')
        for i in tqdm(industry_name):
            industry_ic = close.mask(industry[industry == i].isnull()).corrwith(df,axis = 1,method='spearman').dropna(how = 'all')
            industry_ic_analyse[i] = [industry_ic.mean(),industry_ic.mean()/industry_ic.std()]

        industry_ic_analyse = pd.DataFrame(industry_ic_analyse).T
        industry_ic_analyse.columns = ['IC','IR']
        # 行业IC/IR
        plt.subplot(2, 2, 3)
        plt.bar(industry_ic_analyse.index,industry_ic_analyse.IC)
        plt.xticks(rotation=90)
        plt.title(f'{name}_citics_industry_ic')

    plt.suptitle(f'{name}_ic_analyse')
    plt.show()
    
    return x,half_life,industry_ic_analyse



#### 分层效应
def group_g(df,name = '',n = 20,g = 10):

    """
    :param df: 因子值 -> unstack
    :param df: 因子名称 -> str
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
        return_1d = pd.read_pickle(f'return_1d_{start}_{end}.pkl')
    except:
        return_1d = get_price(order_book_ids, start_date=start, end_date=end,frequency='1d',fields='close').close.unstack('order_book_id').pct_change().shift(-1).dropna(axis = 0,how = 'all').stack()
        return_1d.to_pickle(f'return_1d_{start}_{end}.pkl')

    group = pd.concat([df.stack(),return_1d],axis = 1).dropna()
    group.reset_index(inplace = True)
    group.columns = ['date','stock','factor','current_renturn']

    turnover_ratio = pd.DataFrame()
    group_return = pd.DataFrame()

    for i in tqdm(range(0,len(datetime_period),n)):
        # 调仓
        single = group[group.date == datetime_period[i]].sort_values(by = 'factor')
        
        # 分组
        try:
            single.loc[:,'group'] = pd.qcut(single.factor,g, list(range(1,g+1))).to_list()  # N 组数
        except:
            single = single.replace(0,np.nan).sort_values(by = 'factor')
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

    plt.figure(figsize=(12,12))
    plt.subplot(2, 1, 1)
    plt.plot((group_return+1).cumprod())
    plt.title(f'{name}_group_cum_return')

    plt.subplot(2, 1, 2)
    excess_group_return = (((group_return+1).cumprod().iloc[-1] ** (252/group_return.shape[0]) -1) - ((group_return+1).cumprod().iloc[-1] ** (252/group_return.shape[0]) -1).mean()).to_frame('excess_return')
    plt.bar(excess_group_return.index,excess_group_return.excess_return)
    plt.title(f'{name}_group_return')

    group_return_log = np.log((group_return+1).cumprod()).diff().resample('Y').sum()
    group_return_year = group_return_log.resample('Y').sum().sub(group_return_log.resample('Y').sum().mean(axis = 1),axis = 0).T
    
    return group_return,group_return_year,turnover_ratio