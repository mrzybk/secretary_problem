# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:31:55 2020

@author: lackent
"""

def hire_by_given_rule(n,D,rel_ranks):
    for i in range(n):
        y=rel_ranks[i]
        if D[i][y-1]:
            return i
    return n-1

def evaluate_the_hiring_strategy(n,U,D,CD):
    n=len(D)
    m,counter=0,0
    for c in range(len(CD)):
        a,b=CD[c]
        j=hire_by_given_rule(n,D,a)
        m+=U[b[j]-1]
        counter+=1
    if counter>0:
        return m/counter
    else:
        return 0


