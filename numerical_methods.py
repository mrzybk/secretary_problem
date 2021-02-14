import classical_methods_functions as cmf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    db_address=r"secretary_problem_db.db"
except:
    print('couldnt connect to database')
    pass

import sqlite3
conn = sqlite3.connect(db_address)
cur=conn.cursor()

from sklearn.linear_model import LinearRegression

e=np.exp(1)

def find_threshold_interval(D,i):
    '''
    D: Decisions
    i: relative rank
    
    Returns:
    a,b: accept if the relative rank is i and after a items observed and before b+1 items observed
    '''
    i-=1
    a=i
    if_alpha_found=False
    for d in D:
        #print(not d[i])
        if len(d)>i:
            if not if_alpha_found:
                if not d[i]:
                    a+=1
                else:
                    if_alpha_found=True
                    b=a
            else:
                if d[i]:
                    b+=1
    if b==len(d)-1:
        b+=1
    if a==len(d)-1:
        a+=1
    return a,b

def alpha_threshold_finder_for_multiple_N(Ns,problem,k,s):
    '''this function finds the threshold alphas for the given problem for all N, for the relative rank s'''
    i=s-1
    y_a,y_pol=[],[]
    y_u,y_l=[],[]
    y_aN,y_polN=[],[]
    
    
    q="select * from alphas where problem = {} and k={} and s={} and q is not null".format(problem,k,s)
    df=pd.read_sql_query(q,conn)
    
    
    
    for N in Ns:
        dfs=df[df.n==N]
        if len(dfs)>0:
            
            a=dfs.iloc[0]['alpha']
            
            #print('from database:',a,N)
            y_a.append(a/N)
            y_aN.append(a)
            y_u.append((a+0.5)/N)
            y_l.append((a-0.5)/N)
            y_pol.append(dfs.iloc[0]['q'])
            y_polN.append(dfs.iloc[0]['q']*N)
            if a==N:
                return y_a,y_l,y_u,y_pol,y_aN,y_polN
            
        else:
            Mb=[] 
            U=cmf.create_payoff_array(N,problem,k)
            #print(U)
            D,E,dyn_hiring_rule=cmf.dynamic_solution_general_form(N,U)
            a,b=find_threshold_interval(D,i+1)
            a
            Mb.append(E[0][0])
            if a<N:
                D[a-1][i]=True
                D,E,dyn_hiring_rule=cmf.evaluate_the_strategy(N,U,D)
                Mb.append(E[0][0])
                D[a-1][i]=False


                D[a][i]=False
                D,E,dyn_hiring_rule=cmf.evaluate_the_strategy(N,U,D)
                Mb.append(E[0][0])
                D[a][i]=True


                Ma=[[(a/N)**2,a/N,1],[((a-1)/N)**2,(a-1)/N,1],[((a+1)/N)**2,(a+1)/N,1]]

                qcoefs=np.linalg.solve(Ma,Mb)

                y_a.append(a/N)
                y_aN.append(a)

                y_u.append((a+0.5)/N)
                y_l.append((a-0.5)/N)

                y_pol.append(-(qcoefs[1]/qcoefs[0]/2))
                y_polN.append(-(qcoefs[1]/qcoefs[0]/2)*N)


                cur.execute("insert or ignore into alphas (problem,k,s,n,alpha,q,p_alpha,p_alpha_minus_1,p_alpha_plus_1)\
                            values ('{}','{}','{}','{}','{}','{}','{}','{}','{}')".format(problem,k,s,N,a,-(qcoefs[1]/qcoefs[0]/2),Mb[0],Mb[1],Mb[2]))
                conn.commit()


            else:
                for N0 in Ns:
                    if N0>=N:
                        y_a.append(b/N)
                        y_aN.append(b)
                        y_u.append((b+0.5)/N)
                        y_l.append((b-0.5)/N)
                        y_pol.append(1)
                        y_polN.append(N)
                return y_a,y_l,y_u,y_pol,y_aN,y_polN
        
    return y_a,y_l,y_u,y_pol,y_aN,y_polN


def beta_threshold_finder_for_multiple_N(Ns,problem,k,s):
    '''this function finds the threshold betas for the given problem for all N, for the relative rank s'''
    i=s-1
    y_a,y_pol=[],[]
    y_u,y_l=[],[]
    y_aN,y_polN=[],[]
    
    
    q="select * from betas where problem = {} and k={} and s={} and q is not null".format(problem,k,s)
    df=pd.read_sql_query(q,conn)
    
    
    
    
    
    
    for N in Ns:
        dfs=df[df.n==N]
        if len(dfs)>0:
            
            a=dfs.iloc[0]['beta']
            
            #print('from database:',a,N)
            y_a.append(a/N)
            y_aN.append(a)
            y_u.append((a+0.5)/N)
            y_l.append((a-0.5)/N)
            y_pol.append(dfs.iloc[0]['q'])
            y_polN.append(dfs.iloc[0]['q']*N)
            if a==N:
                return y_a,y_l,y_u,y_pol,y_aN,y_polN
        
        
        else:
            Mb=[] 
            U=cmf.create_payoff_array(N,problem,k)
            #print(U)
            D,E,dyn_hiring_rule=cmf.dynamic_solution_general_form(N,U)
            a,b=find_threshold_interval(D,i+1)
            if b<N:
                Mb=[] 
                U=cmf.create_payoff_array(N,problem,k)
                #print(U)
                D,E,dyn_hiring_rule=cmf.dynamic_solution_general_form(N,U)
                a,b=find_threshold_interval(D,i+1)
                Mb.append(E[0][0])

                D[b-1][i]=False
                D,E,dyn_hiring_rule=cmf.evaluate_the_strategy(N,U,D)
                Mb.append(E[0][0])
                D[b-1][i]=True


                D[b][i]=True
                D,E,dyn_hiring_rule=cmf.evaluate_the_strategy(N,U,D)
                Mb.append(E[0][0])
                D[b][i]=False


                Ma=[[(b/N)**2,b/N,1],[((b-1)/N)**2,(b-1)/N,1],[((b+1)/N)**2,(b+1)/N,1]]

                qcoefs=np.linalg.solve(Ma,Mb)

                y_a.append(b/N)
                y_aN.append(b)

                y_u.append((b+0.5)/N)
                y_l.append((b-0.5)/N)

                y_pol.append(-(qcoefs[1]/qcoefs[0]/2))
                y_polN.append(-(qcoefs[1]/qcoefs[0]/2)*N)
                
                
                
                cur.execute("insert or ignore into betas (problem,k,s,n,beta,q,p_beta,p_beta_minus_1,p_beta_plus_1)\
                            values ('{}','{}','{}','{}','{}','{}','{}','{}','{}')".format(problem,k,s,N,b,-(qcoefs[1]/qcoefs[0]/2),Mb[0],Mb[1],Mb[2]))
                conn.commit()
            else:
                for N0 in Ns:
                    if N0>=N:
                        y_a.append(b/N)
                        y_aN.append(b)
                        y_u.append((b+0.5)/N)
                        y_l.append((b-0.5)/N)
                        y_pol.append(1)
                        y_polN.append(N)
                return y_a,y_l,y_u,y_pol,y_aN,y_polN
        
    return y_a,y_l,y_u,y_pol,y_aN,y_polN


def extrapolate_the_thresholds(Ns,y_polN_a,func_family='f1'):
    clf = LinearRegression(fit_intercept=True)
    if len(y_polN_a)<3:
        return 0,0,0
    if y_polN_a[-1]==Ns[-1]:
        return 0,0,0
    if func_family=='f1':
        clf.fit([[i] for i in Ns],y_polN_a)
    
        b=clf.predict([[0]])[0]
        a=clf.predict([[1]])[0]-b
        return a,b,0
    elif func_family=='f2':
        clf.fit([[i,1/i] for i in Ns],y_polN_a)
        b=clf.predict([[0,0]])[0]
        a=clf.predict([[1,0]])[0]-b
        c=clf.predict([[0,1]])[0]-b
        return a,b,c


    
def calculate_optimal_payoffs_for_multiple_N(Ns,problem,k):
    q="select * from optimal_payoffs \
where problem = {} and k = {}".format(problem,k)
    df=pd.read_sql_query(q,conn)
    for N in Ns:
        dfs=df[df.n==N]
        if len(dfs)==0:
            U=cmf.create_payoff_array(N,problem,k)
            #print(U)
            D,E,dyn_hiring_rule=cmf.dynamic_solution_general_form(N,U)

            cur.execute("insert or ignore into optimal_payoffs (problem,k,n,p_alpha)\
                                        values ('{}','{}','{}','{}')".format(problem,k,N,E[0][0]))
    conn.commit()
    
def from_table_alphas_to_optimal_payoffs():
    q="insert or ignore into optimal_payoffs (problem,k,n,p_alpha) \
select problem,k,n,avg(p_alpha) as p_alpha from alphas \
group by problem,k,n"
    cur.execute(q)
    conn.commit()
    
def plot_winning_probabilities(Ns,problem,k):
    q="select n,p_alpha p from optimal_payoffs \
    where problem = {} and k = {} \
    order by n".format(problem,k)
    df=pd.read_sql_query(q,conn)
    df=df[df['n'].isin(Ns)]
    
    x=[i for i in df['n']]
    y=[np.abs(i) for i in df['p']]
    plt.plot(x,y)
    
    plt.xlabel('n')
    plt.ylabel('P*(n)')
    
    
def print_all_nontrivial_thresholds(Ns,problem,k,rou=6,func_family='f1'):
    my_dict={}
    if k>0:
        s_range=range(1,k+1)
    else:
        s_range=range(1,6)
    #for s in range(1,k+1):
    for s in s_range:
        y_a,y_l,y_u,y_pol_a,y_aN,y_polN_a=alpha_threshold_finder_for_multiple_N(Ns,problem,k,s)
        a,b,c=extrapolate_the_thresholds(Ns,y_polN_a,func_family=func_family)
        if a>0 and a<1:
            #a,b,c=np.round(a,rou),np.round(b,rou),np.round(c,rou)
            #a,b,c='%.6f' %a,'%.6f' %b,'%.6f' %c
            if func_family=='f1':
                f='{}n {} {}'.format('%.6f' %a,'-' if b < 0 else '+','%.6f' %np.abs(b))
            elif func_family=='f2':
                f='{}n {} {} {} {}/n'.format('%.6f' %a,'-' if b < 0 else '+','%.6f' %np.abs(b),
                                             '-' if c < 0 else '+','%.6f' %np.abs(c))
            my_dict[('alpha_{}'.format(s),func_family)]=f
            #print('{} & alpha_{} & {}'.format(k,s,f))
        y_a,y_l,y_u,y_pol_a,y_aN,y_polN_a=beta_threshold_finder_for_multiple_N(Ns,problem,k,s)
        a,b,c=extrapolate_the_thresholds(Ns,y_polN_a,func_family=func_family)
        if a>0 and a<1:
            #a,b,c='%.6f' %a,'%.6f' %b,'%.6f' %c
            if func_family=='f1':
                f='{}n {} {}'.format('%.6f' %a,'-' if b < 0 else '+','%.6f' %np.abs(b))
            elif func_family=='f2':
                f='{}n {} {} {} {}/n'.format('%.6f' %a,'-' if b < 0 else '+','%.6f' %np.abs(b),'-' if c < 0 else '+','%.6f' %np.abs(c))
            my_dict[('beta_{}'.format(s),func_family)]=f
            #print('{} & beta_{} & {}'.format(k,s,f))
    return my_dict


def get_theory_solution(problem,k,s,if_alpha=True):
    if if_alpha:
        if problem==2 and k==3 and s==3:
            theory_solution_alpha=1/np.sqrt(e)
        elif problem==2 and k==3 and s==2:
            theory_solution_alpha=2/(2*np.sqrt(e)+np.sqrt(4*e-6*np.sqrt(e)))
        elif problem==2 and k==2 and s==2:
            theory_solution_alpha=0.5
        elif problem<=3 and k==1 and s==1:
            theory_solution_alpha=1/e
        elif problem==3 and k==2 and s==1:
            theory_solution_alpha=0.3469816097075797772 #gilbert1966 solution of e^(x-1)=2x/3
        elif problem==3 and k==2 and s==2:
            theory_solution_alpha=2/3
        elif problem == 1 and s==1:
            theory_solution_alpha=0.2584
        elif problem == 1 and s==2:
            theory_solution_alpha=0.4476
        elif problem == 1 and s==3:
            theory_solution_alpha=0.5640
        else:
            theory_solution_alpha=0
        return theory_solution_alpha
        
    else:
        return 0

def alpha_threshold_finder_for_multiple_N_bkp(Ns,problem,k,s):
    '''this function finds the threshold alphas for the given problem for all N, for the relative rank s'''
    i=s-1
    y_a,y_pol=[],[]
    y_u,y_l=[],[]
    y_aN,y_polN=[],[]
    for N in Ns:
        
        Mb=[] 
        U=cmf.create_payoff_array(N,problem,k)
        #print(U)
        D,E,dyn_hiring_rule=cmf.dynamic_solution_general_form(N,U)
        a,b=find_threshold_interval(D,i+1)
        a
        Mb.append(E[0][0])
        if a<N:
            D[a-1][i]=True
            D,E,dyn_hiring_rule=cmf.evaluate_the_strategy(N,U,D)
            Mb.append(E[0][0])
            D[a-1][i]=False


            D[a][i]=False
            D,E,dyn_hiring_rule=cmf.evaluate_the_strategy(N,U,D)
            Mb.append(E[0][0])
            D[a][i]=True


            Ma=[[(a/N)**2,a/N,1],[((a-1)/N)**2,(a-1)/N,1],[((a+1)/N)**2,(a+1)/N,1]]

            qcoefs=np.linalg.solve(Ma,Mb)

            y_a.append(a/N)
            y_aN.append(a)

            y_u.append((a+0.5)/N)
            y_l.append((a-0.5)/N)

            y_pol.append(-(qcoefs[1]/qcoefs[0]/2))
            y_polN.append(-(qcoefs[1]/qcoefs[0]/2)*N)
        else:
            for N0 in Ns:
                y_a.append(b/N)
                y_aN.append(b)
                y_u.append((b+0.5)/N)
                y_l.append((b-0.5)/N)
                y_pol.append(1)
                y_polN.append(N)
            return y_a,y_l,y_u,y_pol,y_aN,y_polN
        
    return y_a,y_l,y_u,y_pol,y_aN,y_polN

def beta_threshold_finder_for_multiple_N_bkp(Ns,problem,k,s):
    '''this function finds the threshold betas for the given problem for all N, for the relative rank s'''
    i=s-1
    y_a,y_pol=[],[]
    y_u,y_l=[],[]
    y_aN,y_polN=[],[]
    for N in Ns:
        Mb=[] 
        U=cmf.create_payoff_array(N,problem,k)
        #print(U)
        D,E,dyn_hiring_rule=cmf.dynamic_solution_general_form(N,U)
        a,b=find_threshold_interval(D,i+1)
        if b<N:
            Mb=[] 
            U=cmf.create_payoff_array(N,problem,k)
            #print(U)
            D,E,dyn_hiring_rule=cmf.dynamic_solution_general_form(N,U)
            a,b=find_threshold_interval(D,i+1)
            Mb.append(E[0][0])

            D[b-1][i]=False
            D,E,dyn_hiring_rule=cmf.evaluate_the_strategy(N,U,D)
            Mb.append(E[0][0])
            D[b-1][i]=True


            D[b][i]=True
            D,E,dyn_hiring_rule=cmf.evaluate_the_strategy(N,U,D)
            Mb.append(E[0][0])
            D[b][i]=False


            Ma=[[(b/N)**2,b/N,1],[((b-1)/N)**2,(b-1)/N,1],[((b+1)/N)**2,(b+1)/N,1]]

            qcoefs=np.linalg.solve(Ma,Mb)

            y_a.append(b/N)
            y_aN.append(b)

            y_u.append((b+0.5)/N)
            y_l.append((b-0.5)/N)

            y_pol.append(-(qcoefs[1]/qcoefs[0]/2))
            y_polN.append(-(qcoefs[1]/qcoefs[0]/2)*N)
        else:
            for N0 in Ns:
                y_a.append(b/N)
                y_aN.append(b)
                y_u.append((b+0.5)/N)
                y_l.append((b-0.5)/N)
                y_pol.append(1)
                y_polN.append(N)
            return y_a,y_l,y_u,y_pol,y_aN,y_polN
        
    return y_a,y_l,y_u,y_pol,y_aN,y_polN

