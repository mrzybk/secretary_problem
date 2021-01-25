import classical_methods_functions as cmf
import numpy as np
import pandas as pd
db_address=r"secretary_problem_db.db"

import sqlite3
conn = sqlite3.connect(db_address)
cur=conn.cursor()

from sklearn.linear_model import LinearRegression

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

