# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:31:55 2020

@author: lackent
"""

import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta

def generate_candidates(n):
    '''
    Creates a list of candidates with their rank, and also outputs the relative ranks
    
    Parameters:
    n: number of candidates
    
    Returns:
    rel_ranks: relative ranks of the candidates
    b: ranks of the candidates
    '''
    b=np.arange(1,n+1)
    np.random.shuffle(b)
    rel_ranks=[1 for i in range(n)]
    for j in range(1,n):
        r=1
        for i in b[:j]:
            if b[j]>i:
                r+=1
        rel_ranks[j]=r
    return rel_ranks,[i for i in b]

def generate_candidates_v2(n):
    '''
    Creates a list of candidates with their rank, and also outputs the relative ranks
    
    Unlike the first version it randomly generates the relative ranks, then calculate the actual ranks. 
    That's why it is much faster for n>100
    
    Parameters:
    n: number of candidates
    
    Returns:
    rel_ranks: relative ranks of the candidates
    b: ranks of the candidates
    '''
    a=[i for i in range(n)]
    rel_ranks=[np.random.randint(0,j+1) for j in range(n)]
    b=[0 for i in range(n)]
    for i in range(n-1,-1,-1):
        b[i]=a[rel_ranks[i]]+1
        del a[rel_ranks[i]]
    return [r+1 for r in rel_ranks],b


def generate_single_data_from_the_stock_data(n,last_data_date,df):
    df['val']=df.iloc[:,1].apply(lambda x:float(x.split('$')[1]))
    b_s=[v for v in df[df.Date<last_data_date].iloc[0:n].sort_values(by='Date',ascending=True).val]
    
    #find the relative ranks out of the values
    rel_ranks=[0 for i in range(n)]
    for j in range(1,n):
        r=1
        for i in b_s[:j]:
            if b_s[j]>i:
                r+=1
        rel_ranks[j]=r-1

    #using relative ranks find the real ranks
    a=[i for i in range(n)]
    b=[0 for i in range(n)]
    for i in range(n-1,-1,-1):
        b[i]=a[rel_ranks[i]]+1
        del a[rel_ranks[i]]


    return [r+1 for r in rel_ranks],b,b_s

def generate_multi_data_from_the_stock_data(n,m,interv,last_data_date,file_name):
    df=pd.read_csv('data/'+file_name+'.csv',parse_dates=[0,1])
    CD=[]
    for i in range(m):
        r,b,b_s=generate_single_data_from_the_stock_data(n,last_data_date,df)
        CD.append([r,b,b_s])
        last_data_date=dt.strftime(dt.strptime(last_data_date,'%Y-%m-%d')-timedelta(days=interv),'%Y-%m-%d')
    return CD
