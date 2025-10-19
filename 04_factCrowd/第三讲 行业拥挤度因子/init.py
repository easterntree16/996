# 数据分析
import pandas as pd 
import numpy as np
import pickle
from tqdm import *
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import scipy.stats as st
from scipy import stats
import statsmodels.api as sm
from scipy import stats

# 米筐
from rqdatac import *
init()
import rqdatac
from rqfactor_utils import *
from rqfactor_utils.universe_filter import *
from rqfactor import *
from rqfactor.extension import UserDefinedLeafFactor
from rqfactor import REF

# 关闭通知
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 动态券池
def INDEX_FIX(start_date = '2016-02-01',end_date = '2023-10-01',index_item = '000906.XSHG'):
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
    stock_list = index_fix.columns.tolist()

    return index_fix,stock_list

def get_industry_exposure(order_book_ids,datetime_period,type = 'zx'):
    
    """
    :param order_book_ids: 股票池 -> list
    :param datetime_period: 研究日 -> list
    :return result: 虚拟变量 -> dataframe
    """
    print('gen industry martix... ')
    if type != 'zx':
        industry = rqdatac.client.get_client().execute('__internal__shenwan_industry')
        df = pd.DataFrame(industry)
        df.set_index(['order_book_id', 'start_date'], inplace=True)
        df = df['index_name'].sort_index()
    else:
        industry = rqdatac.client.get_client().execute('__internal__zx2019_industry')
        df = pd.DataFrame(industry)
        df.set_index(['order_book_id', 'start_date'], inplace=True)
        df = df['first_industry_name'].sort_index()
    
    
    #构建动态行业数据表格
    index = pd.MultiIndex.from_product([order_book_ids, datetime_period], names=['order_book_id', 'date'])
    pos = df.index.searchsorted(index, side='right') - 1
    index = index.swaplevel()   # level change (oid, datetime) --> (datetime, oid)
    result = pd.Series(df.values[pos], index=index)
    result = result.sort_index()
    
    #生成行业虚拟变量
    return result

