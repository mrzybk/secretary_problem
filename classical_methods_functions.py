# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:31:55 2020

@author: lackent
"""

import numpy as np
import scipy.stats as ss
import scipy as sp

import simple_math_functions as smf

def create_payoff_array(n,problem,k):
    '''
    
    problems:
    -pick_the_best:0
    -maximize_expected_rank:1
    -pick_exactly_the_kth_best:2
    -pick_from_the_top_k_best:3
    -pick_odds_or_evens_depending_on_k:4
    -pick_primes:5
    '''
    if problem==0:
        return [1 if i==1 else 0 for i in range(1,n+1)]
    elif problem==1:
        return [-i for i in range(1,n+1)]
    elif problem==2:
        return [1 if i==k else 0 for i in range(1,n+1)]
    elif problem==3:
        return [1 if i<=k else 0 for i in range(1,n+1)]
    elif problem==4:
        return [1 if i%2==k%2 else 0 for i in range(1,n+1)]
    elif problem==5:
        return [1 if smf.is_prime(i) else 0 for i in range(1,n+1)]
def prob_of(n,i,r,s):
    ''' This function returns the probability that the true rank out of all n items is i given that 
        the observer chooses to stop in state (r, s) and accept the item that is the sth best out of the r items
        
        Reference: The Secretary Problem and its Extensions: A Review P.R. Freeman (1983) (3.1)'''
    v1=sp.misc.comb(n-i,r-s)
    v2=sp.misc.comb(i-1,s-1)
    v3=sp.misc.comb(n,r)
    return v1*v2/v3


def calculate_payoff_expected_rank_minimization(n,r,s):
    '''there are n candidates, r observed and the rank of rth is s out of r people when the candidate is expected
    '''
    probs=[]

    for i in range(1,n+1):
        probs.append(prob_of(n,i,r,s))
    return sum([i*probs[i-1] for i in range(1,n+1)])

def dynamic_solution_expected_rank_minimization(n):
    '''
        Returns in each situtaion whether to accept or continue as matrix. 
        Decisions_Accept(a,b) is a boolean that suggests when the relative rank is a+1 of bth candidate  to accept or not.
    '''
    Possible_Outcomes=[[i+1 for i in range(j)] for j in range(1,n+1)]
    Decisions_Accept=[[True for i in range(j)] for j in range(1,n+1)]
    Expecteds=[[0 for i in range(j)] for j in range(1,n)]+[[i for i in range(1,n+1)]]
    
    e=len(Expecteds)-2
    for i in range(e,-1,-1):
        print(i)
        arr_0=Possible_Outcomes[i]
        arr_1=Expecteds[i]
        a=len(arr_0)
        for b in Possible_Outcomes[i]:
            expected_a=calculate_payoff_expected_rank_minimization(n,a,b) #accept
            expected_r=sum(Expecteds[i+1])/(a+1) #reject
            if expected_r<expected_a: #if rejecting is better
                Decisions_Accept[i][b-1]=False
            Expecteds[i][b-1]=min(expected_a,expected_r)
    return Decisions_Accept,Expecteds,[sum(d) for d in Decisions_Accept]

def dynamic_solution_exactly_kth_best(n,k):
    Possible_Outcomes=[[i+1 for i in range(j)] for j in range(1,n+1)]
    Decisions_Accept=[[True for i in range(j)] for j in range(1,n+1)]
    Expecteds=[[0 for i in range(j)] for j in range(1,n)]+[[1 if i==k else 0 for i in range(1,n+1)]]

    e=len(Expecteds)-2
    for i in range(e,-1,-1):
        print(i)
        arr_0=Possible_Outcomes[i]
        arr_1=Expecteds[i]
        a=len(arr_0)
        for b in Possible_Outcomes[i]:
            expected_a=prob_of(n,k,a,b) #accept
            expected_r=sum(Expecteds[i+1])/(a+1) #reject
            if expected_r>expected_a: #rejecting is better
                Decisions_Accept[i][b-1]=False
            Expecteds[i][b-1]=max(expected_a,expected_r)
    return Decisions_Accept,Expecteds,[sum(d) for d in Decisions_Accept]

def dynamic_solution_from_the_top_k_best(n,k):
    Possible_Outcomes=[[i+1 for i in range(j)] for j in range(1,n+1)]
    Decisions_Accept=[[True for i in range(j)] for j in range(1,n+1)]
    Expecteds=[[0 for i in range(j)] for j in range(1,n)]+[[1 if i<=k else 0 for i in range(1,n+1)]]

    e=len(Expecteds)-2
    for i in range(e,-1,-1):
        print(i)
        arr_0=Possible_Outcomes[i]
        arr_1=Expecteds[i]
        a=len(arr_0)
        for b in Possible_Outcomes[i]:
            expected_a=sum([prob_of(n,j,a,b) for j in range(1,k+1)]) #accept
            expected_r=sum(Expecteds[i+1])/(a+1) #reject
            if expected_r>expected_a: #rejecting is better
                Decisions_Accept[i][b-1]=False
            Expecteds[i][b-1]=max(expected_a,expected_r)
    return Decisions_Accept,Expecteds,[sum(d) for d in Decisions_Accept]

def dynamic_solution_general_form(n,U):
    '''n: the number of candidates
       U: payoffs for each candidate, U[x-1] is the payoff of best xth candidate. 
       
       returns the strategy which maximizes the expected payoff
       
       For example: U=[1,0,0,.....] gives the classical problem. U=[-1,-2,..,-n] gives the expected rank minimization
    '''
    
    
    Possible_Outcomes=[[i+1 for i in range(j)] for j in range(1,n+1)]
    Decisions_Accept=[[True for i in range(j)] for j in range(1,n+1)]
    Expecteds=[[0 for i in range(j)] for j in range(1,n)]+[U]

    e=len(Expecteds)-2
    for i in range(e,-1,-1):
        print(i)
        arr_0=Possible_Outcomes[i]
        arr_1=Expecteds[i]
        a=len(arr_0)
        expected_r=sum(Expecteds[i+1])/(a+1) #reject
        for b in Possible_Outcomes[i]:
            expected_a=sum([prob_of(n,j,a,b)*U[j-1] for j in range(1,n+1)]) #accept
            
            if expected_r>expected_a: #rejecting is better
                Decisions_Accept[i][b-1]=False
            Expecteds[i][b-1]=max(expected_a,expected_r)
    return Decisions_Accept,Expecteds,[sum(d) for d in Decisions_Accept]
