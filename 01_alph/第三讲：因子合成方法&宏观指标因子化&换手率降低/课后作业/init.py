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
index_item = '000906.XSHG'            # 【example】：全A 000985.XSHG 中证1000 000852.XSHG

# 股票池
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

index_fix = INDEX_FIX(start_date,end_date,index_item)

stock_list = index_fix.columns.tolist()
date_list = index_fix.index.tolist()


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
