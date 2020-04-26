# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:31:55 2020

@author: lackent
"""

import numpy as np

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
