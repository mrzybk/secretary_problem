# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:31:55 2020

@author: lackent
"""
import numpy as np


def hire_by_given_rule(n,D,rel_ranks):
    if len(D)==0:
        return np.random.randint(0,n) #hire one candidate uniformly random
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



def evaluate_the_hiring_strategy_on_stocks(n,U,D,CD):
    m,counter=0,0
    paid_money=0
    received_money=0
    for c in range(len(CD)):
        a,b,p=CD[c][0],CD[c][1],CD[c][2]
        j=hire_by_given_rule(n,D,a)
        m+=U[b[j]-1]
        counter+=1
        print(b[j],p[j],p[-1],a,b,p)
        paid_money+=p[j]
        received_money+=p[-1]
    if counter>0:
        return m/counter,paid_money,received_money,received_money/paid_money-1
    else:
        return 0