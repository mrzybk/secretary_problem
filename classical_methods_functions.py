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
    -minimize_expected_rank:1
    -minimize_expected_square_rank:1.1
    -minimize_expected_cube_rank:1.13
    -minimize_expected_exponential_rank:1.2
    -minimize_expected_distance_to_rank_k:1.5
    -minimize_expected_squared_distance_to_rank_k:1.51
    -minimize_expected_distance_to_rank_median:1.53
    -minimize_expected_squared_distance_to_rank_median:1.54
    -pick_around_median_rose1982:1.6
    -pick_exactly_the_kth_best:2
    -pick_from_the_top_k_best:3
    -pick_odds_or_evens_depending_on_k:4
    -pick_primes:5
    -pick_primes_except_the_last_k:5.1
    -pick_primes_or_from_the_worst_k:5.2
    -pick_primes_from_the_first_half:5.3
    -pick_primes_one_of_top_k_primes:5.4
    -pick_from_the_best_k_or_the_worst_k:6
    -pick_the_divisors_of_N:9
    '''
    if problem==0:
        return [1 if i==1 else 0 for i in range(1,n+1)]
    elif problem==1:
        return [-i for i in range(1,n+1)]
    elif problem==1.1:
        return [-i**2 for i in range(1,n+1)]
    elif problem==1.13:
        return [-i**3 for i in range(1,n+1)]
    elif problem==1.2:
        return [-np.exp(i) for i in range(1,n+1)]
    elif problem==1.5:
        return [-np.abs(i-k) for i in range(1,n+1)]
    elif problem==1.51:
        return [-(i-k)**2 for i in range(1,n+1)]
    elif problem==1.52:
        k=(n+1)/2
        return [-np.abs(i-k) for i in range(1,n+1)]
    elif problem==1.53:
        k=(n+1)/2
        return [-(i-k)**2 for i in range(1,n+1)]
    elif problem==1.6:
        med=(n+1)/2
        return [1 if i>=med-med**k/2 and i<=med+med**k/2 else 0 for i in range(1,n+1)]
    elif problem==2:
        return [1 if i==k else 0 for i in range(1,n+1)]
    elif problem==3:
        return [1 if i<=k else 0 for i in range(1,n+1)]
    elif problem==4:
        return [1 if i%2==k%2 else 0 for i in range(1,n+1)]
    elif problem==5:
        return [1 if smf.is_prime(i) else 0 for i in range(1,n+1)]
    elif problem==5.1:
        U=[1 if smf.is_prime(i) else 0 for i in range(1,n+1)]
        i=-1
        for j in range(k):
            while U[i]==0 and i>-n:
                i-=1
            U[i]=0
        return U
    elif problem==5.2:
        return [1 if smf.is_prime(i) else 0 for i in range(1,n+1-k)]+[1 for i in range(k)]
    elif problem==5.3:
        return [1 if smf.is_prime(i) and i<n/2 else 0 for i in range(1,n+1)]
    elif problem==5.4:
        U=[1 if smf.is_prime(i) else 0 for i in range(1,n+1)]
        c=0
        for i in range(len(U)):
            if c>=k:
                U[i]=0
            if U[i]==1:
                c+=1
        return U
            
    elif problem==6:
        return [1 if i<=k or i>n-k else 0 for i in range(1,n+1)]
    elif problem==9:
        return [1 if n%i==0 else 0 for i in range(1,n+1)]
    
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

def dynamic_solution_general_form_archived(n,U):
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
        #print(i)
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

def calculate_the_payoff_when_item_accepted(n,U):
    I=np.zeros((n,n))
    I[-1]=U.copy()
    for r in range(-2,-n-1,-1):
        for s in range(0,n+r+1):
            #I[r][s]=I[r+1][s]*(1-(s+1)/(n+r+2))+I[r+1][s+1]*((s+1)/(n+r+2))
            I[r][s]=(I[r+1][s]*(n+r+1-s)+I[r+1][s+1]*(s+1))/(n+r+2)
    return I
    

def dynamic_solution_general_form(n,U):
    '''n: the number of candidates
       U: payoffs for each candidate, U[x-1] is the payoff of best xth candidate. 
       
       returns the strategy which maximizes the expected payoff
       
       For example: U=[1,0,0,.....] gives the classical problem. U=[-1,-2,..,-n] gives the expected rank minimization
    '''
    Decisions_Accept=[[True for i in range(j)] for j in range(1,n+1)]
    I=calculate_the_payoff_when_item_accepted(n,U)

    E=np.zeros((n,n))
    E[-1]=U.copy()
    for r in range(-2,-n-1,-1):
        expected_r=sum(E[r+1])/(n+r+2)
        for s in range(0,n+r+1):
            if I[r][s]<expected_r:
                Decisions_Accept[r][s]=False
                E[r][s]=expected_r
            else:
                E[r][s]=I[r][s]

    return Decisions_Accept,E,[sum(d) for d in Decisions_Accept]



def find_expected_nr_of_observed_candidates(N,D):
    '''This functions finds the expected number of observed candidates with the strategy D'''
    e=0 #nr of expected steps
    p_cont=1 #prob of going to the ith stage (initial 1)
    for i in range(1,N+1):
        p_stop=sum(D[i-1])/len(D[i-1])*p_cont #prob of stopping at the ith candidate
        e+=i*p_stop
        p_cont-=p_stop
        #print(i,p_stop,e/N)
    return e


def evaluate_the_strategy(n,U,Decisions_Accept):
    ####TODO Update this
    I=calculate_the_payoff_when_item_accepted(n,U)

    E=np.zeros((n,n))
    E[-1]=U.copy()
    for r in range(-2,-n-1,-1):
        expected_r=sum(E[r+1])/(n+r+2)
        for s in range(0,n+r+1):
            if Decisions_Accept[r][s]:
                E[r][s]=I[r][s]
            else:
                E[r][s]=expected_r


    return Decisions_Accept,E,[sum(d) for d in Decisions_Accept]
    return Decisions_Accept,Expecteds,[sum(d) for d in Decisions_Accept]


#classic secretary
def fast_classic_secretary_solver(N):
    '''This function returns the number of items need to be observed before picking the relative best'''
    r=N-1
    summ_=0
    while summ_<1:
        summ_+=1/r
        r-=1
        #print(r+1,summ_)
    return r+1