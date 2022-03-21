#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 13:34:21 2020

@author: honghanye
"""

import numpy as np
import random
from random import seed
from joblib import Parallel, delayed
from scipy.stats import poisson
import csv
from scipy.stats import t
import matplotlib.pyplot as plt

def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

def learn_rank(Distname,p,size):
    if Distname == 'norm':
        X = np.zeros([p,size])
        mean = np.zeros(p)
        covariance = np.identity(p)
        X = np.random.multivariate_normal(mean,covariance,size).T
    
    elif Distname == 'correlated':
        X = np.zeros([p,size])

        mean = 25 *[0]    
        covariance = np.zeros([25,25])
        for i in range(25):
            for j in range(25):
                covariance[i,j] = 0.8** abs(i-j)
        X[0:25,:] = np.random.multivariate_normal(mean,covariance,size).T
        X[25:50,:] = multivariate_t_rvs(mean,covariance,df = 3,n = size).T
        X[25:50,:] = X[25:50,:]/np.sqrt(3)
    
    elif Distname == 'mixed':
        X = np.zeros([p,size])
        
        X[0:10,:] = np.random.normal(0,1,[10,size])
        X[10:20,:] = np.array(t.rvs(3,size = [10,size]))/np.sqrt(3)
        X[20:30,:] = 1 - np.random.exponential(1, [10,size]) 
        X[30:40,:] = (np.random.lognormal(1,0.5, [10,size]) - 3)/1.6 
        
        for j in range(40,50):
            X[j,:] =  (poisson.rvs(10, size = size) - 10)/np.sqrt(10)
         
    rank_min = np.zeros([p,size])
    rank_max = np.zeros([p,size])

    for i in range(size):
        dat = X[:,i]
        tie_index = np.where(dat == dat.min())
        rank_min[tie_index,i] = 1/len(tie_index)
        tie_index_up = np.where(dat == dat.max())
        rank_max[tie_index_up,i] = 1/len(tie_index_up)
    
    prob_min = np.sum(rank_min,axis = 1)/size
    prob_max = np.sum(rank_max,axis = 1)/size
    '''
    plt.bar(np.arange(0,50),prob_min)
    plt.xlabel('Data stream index')
    plt.ylabel('In-control antirank probability')
    plt.savefig('In-contrl_0626.png',dpi = 600)
    '''
    return [prob_min,prob_max]
    

def dirichlet_sample(data):
    
    rnd_dat = np.random.gamma(data,1)
    
    return rnd_dat/sum(rnd_dat)
    

def AiTS(Distname,r,u1,k,h,temporal_loc,g_min, g_max,locc):
    
    p = g_min.shape[0]
    q= 100000    
    
    if Distname == 'norm':
        X = np.zeros([p,q])
        mean = np.zeros(p)
        covariance = np.identity(p)
        X = np.random.multivariate_normal(mean,covariance,q).T
        
        if u1 != 0 :
            loc = 0;#random.sample(range(0,p),locc)
            X[loc,temporal_loc:q] = np.random.normal(0,1,q-temporal_loc) + u1

        
    elif Distname == 'mixed':
        
        X = np.zeros([p,q])
        X[0:10,:] = np.random.normal(0,1,[10,q])
        X[10:20,:] = np.array(t.rvs(3,size = [10,q]))/np.sqrt(3)
        X[20:30,:] = 1 - np.random.exponential(1, [10,q]) 
        X[30:40,:] = (np.random.lognormal(1,0.5, [10,q] ) - 3)/1.6 
        X[40:50,:] = (np.random.poisson(10, [10,q]) - 10)/np.sqrt(10)
        
        if u1 != 0:
          
            lll = random.sample(range(0,p),locc)
            for loc in lll:
                if loc < 10:
                    X[loc, temporal_loc:q] = np.random.normal(0,1,q-temporal_loc) + u1
                elif loc >= 10 and loc < 20:
                    X[loc,temporal_loc:q] = np.array(t.rvs(3,size = q-temporal_loc))/np.sqrt(3) + u1
                elif loc >=20 and loc < 30:
                    X[loc,temporal_loc:q] = 1 - np.random.exponential(1, size = q-temporal_loc)  + u1
                elif loc >= 30 and loc < 40:
                    X[random.randrange(30,40), temporal_loc:q] = (np.random.lognormal(1,0.5, size = q-temporal_loc) - 3)/1.6 + u1
                else:
                    X[random.randrange(40,50),temporal_loc:q] = (np.random.poisson(10, size=q-temporal_loc) - 10)/np.sqrt(10) + u1
                    
    elif Distname == 'correlated':
        X = np.zeros([p,q])

        mean = 25 *[0]    
        covariance = np.zeros([25,25])
        for i in range(25):
            for j in range(25):
                covariance[i,j] = 0.8** abs(i-j)
        X[0:25,:] = np.random.multivariate_normal(mean,covariance,q).T
        X[25:50,:] = multivariate_t_rvs(mean,covariance,df = 3,n = q).T
        X[25:50,:] = X[25:50,:]/np.sqrt(3)
        
        
        if u1 != 0 :
            if locc == 1:
                loc = random.randrange(p)
                mean1 = 50*[0]
                mean1[loc] = u1
                X[0:25,:] = np.random.multivariate_normal(mean1[0:25],covariance,q).T
                X[25:50,:] = multivariate_t_rvs(mean1[25:50],covariance,df = 3,n = q).T
                X[25:50,:] = X[25:50,:]/np.sqrt(3)
              
            elif locc == 0:
                for i in range(25):
                    for j in range(25):
                        covariance[i,j] = (0.8+u1*0.2)** abs(i-j)
                X[0:25,:] = np.random.multivariate_normal(mean,covariance,q).T
                X[25:50,:] = multivariate_t_rvs(mean,covariance,df = 3,n = q).T
                X[25:50,:] = X[25:50,:]/np.sqrt(3)
            else:
                mean1 = 50*[0]
                loc = random.sample(range(0,p),locc)
               
                for lll in loc:
                    mean1[lll] = u1
                X[0:25,:] = np.random.multivariate_normal(mean1[0:25],covariance,q).T
                X[25:50,:] = multivariate_t_rvs(mean1[25:50],covariance,df = 3,n = q).T
                X[25:50,:] = X[25:50,:]/np.sqrt(3)
          
    fulllist = np.arange(0,p,1) 
    observed_down = np.arange(0,r,1)
    unobserved_down = np.setdiff1d(fulllist, observed_down)
    
    S1_down = np.zeros([p,q])    
    S2_down = np.zeros([p,q])
    C_down  = np.zeros(q)
    xi_down = np.zeros([p,q])
    theta_down = np.zeros([p,q])   
    alpha_down = g_min 
    weight_rec = np.zeros([p,q])
    y = np.zeros(q)

    i = 0
    weight_down = np.zeros([p])
    X_data_down = np.copy(X[:,i])
    X_data_down[unobserved_down] = 10000
    
    tie_index = np.where(X_data_down == X_data_down.min())
    weight_down[tie_index] = sum(g_min[observed_down])/len(tie_index)
    
    weight_down[unobserved_down] = g_min[unobserved_down]
    weight_rec[:,i] = weight_down
    alpha_down = alpha_down + weight_down 
    theta_down[:,i] = alpha_down/sum(alpha_down)

    xi_down[tie_index,i] = sum(theta_down[observed_down,i])/len(tie_index)
    xi_down[unobserved_down,i] = theta_down[unobserved_down,i]
    
    C_down[0] = np.matmul(np.matmul((xi_down[:,0]).transpose(), np.diag(1/g_min)), xi_down[:,0] )

    if C_down[0] <= k:
        S1_down[:,0] = S1_down[:,0]
        S2_down[:,0] = S2_down[:,0]
        observed_down = observed_down
        unobserved_down = np.setdiff1d(fulllist, observed_down)
    else:
        S1_down[:,0] = xi_down[:,0] * (1- k/C_down[0]) 
        S2_down[:,0] = g_min * (1-k/C_down[0])
        N = dirichlet_sample(S1_down[:,0]/sum(S1_down[:,0]))
        observed_down =   N.argsort()[::-1][:r]
        unobserved_down = np.setdiff1d(fulllist, observed_down)
    y[0] = max(0, C_down[0] - k)    
    i = 1
    
    while ( i < temporal_loc or  (y[i-1] < h and i <= q - 1)):
       
        weight_down = np.zeros([p])
        X_data_down = np.copy(X[:,i])
        X_data_down[unobserved_down] = 10000
        tie_index = np.where(X_data_down == X_data_down.min())
        weight_down[tie_index] = sum(g_min[observed_down])/len(tie_index)
        weight_down[unobserved_down] = g_min[unobserved_down]
        weight_rec[:,i] = weight_down

        alpha_down = alpha_down + weight_down 
        
        theta_down[:,i] = alpha_down/sum(alpha_down)
        xi_down[tie_index,i] = sum(theta_down[observed_down,i])/len(tie_index)
        xi_down[unobserved_down,i] = theta_down[unobserved_down,i]
        C_down[i] = np.matmul(np.matmul((S1_down[:,i-1]-S2_down[:,i-1]+xi_down[:,i]-g_min).transpose(), np.diag(1/(S2_down[:,i-1]+g_min))), (S1_down[:,i-1]-S2_down[:,i-1]+xi_down[:,i]-g_min) )

        if C_down[i] <= k:
            S1_down[:,i] = S1_down[:,i]
            S2_down[:,i] = S2_down[:,i]
            observed_down = observed_down
            unobserved_down = np.setdiff1d(fulllist, observed_down)
        else:
            S1_down[:,i] = (S1_down[:,i-1] + xi_down[:,i]) * (1- k/C_down[i]) 
            S2_down[:,i] = (S2_down[:,i-1] + g_min ) * (1-k/C_down[i])
            N = dirichlet_sample(S1_down[:,i]/sum(S1_down[:,i]))
            observed_down =   N.argsort()[::-1][:r]
            unobserved_down = np.setdiff1d(fulllist, observed_down)
           
        y[i] = max(0, C_down[i] - k)     
        i = i + 1
    return(weight_rec)

def learn(Distname,r,u1,k,h,temporal_loc,g_min,g_max,loc):
 
    
    T = 10000      
    num_cores = 25 
    
    Y = Parallel(n_jobs = num_cores,max_nbytes= None)(delayed(AiTS)(Distname,r,u1, k, h, temporal_loc,g_min, g_max,loc) for i in range(T))

    Y1 = np.asarray(Y)
    Y1 = Y1[Y1 >= temporal_loc] - temporal_loc + 1
    ARL = np.mean(Y1)
    std_ARL = np.std(Y1)/np.sqrt(len(Y1))  
    
    return [ARL, std_ARL] 


def learnh_ic(Distname,r,k,g_min, g_max):


    RL = 370
    u1 = 0
    small_delta = 2
    h1 = 1
    h2 = 1.2
    
    A,B = learn(Distname, r,u1, k, h2,0,g_min, g_max,0)

    if abs(A-RL) < small_delta:
        print('it is finished')
    

    while A< RL:
        if r == 5:
            h2 = h2 + 0.5
        else:
            h2 = h2 + 3
        A,B = learn(Distname,r, u1, k, h2,0,g_min, g_max,0)
        if abs(A- RL) < small_delta:
            print('it is finished')
    
    h1= h2/2
    h = (h1+h2)/2

    A,B = learn(Distname,r, u1, k, h,0,g_min, g_max,0)

    if abs(A-RL) < small_delta:
        print('it is finished')

    while abs(A-RL)> small_delta and abs(h1-h2) > 0.00001:
        if A < RL:
            h1 = h
            h = (h1+h2)/2
            A,B = learn(Distname,r, u1, k, h,0,g_min, g_max,0)
        else:
            h2 = h
            h = (h1+h2)/2
            A,B = learn(Distname,r, u1, k, h,0,g_min, g_max,0)

    print(h)
    
    output = np.zeros([7,2]) 
    output[0,0] = A
    output[0,1] = B
    indexx = 1
    mean_shift_range = [-1,-2,-3]
    
    for loc in [1,3]:
        for u2 in mean_shift_range:
            A1,B1 =  learn(Distname, r,u2, k, h,0,g_min, g_max,loc)
            output[indexx,0] = A1
            output[indexx,1] = B1
            indexx = indexx + 1
    
    return output

np.random.seed(100000)
random.seed(100000)


Name_range = ['correlated']#,'correlated']
Distname = 'norm'
p = 6
k = 0.1
output = np.zeros([18,2])
index = 0
for samplesize in [10000]:
    [g_min, g_max] = learn_rank(Distname,p,samplesize)
    for r in [15]:
        YY = learnh_ic(Distname,r,k,g_min, g_max)
        print(YY)
        
        with open(str(Distname) + '_AiTS_k_' + str(k) + ' r is ' + str(r) + 'R2.csv', 'w',newline = '') as f:
            writer  = csv.writer(f)
            writer.writerows(YY)
            
g_min = np.ones([6])*1/6
g_max = g_min

a = AiTS(Distname,4,0,k,h,temporal_loc,g_min, g_max,locc)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['1', '2', '3', '4', '5','6']
students =a.mean(1)
ax.bar(langs,students)
ax.set_ylabel('Mean of $\mathbf{\hat{\omega}} (t)$')
ax.set_xlabel('Data streams index')
plt.savefig('in-control.png',bbox_inches='tight',dpi  = 500)




b= AiTS(Distname,4,-1,k,h,temporal_loc,g_min, g_max,locc)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

langs = ['1', '2', '3', '4', '5','6']
students =b.mean(1)
ax.bar(langs,students)
ax.set_ylabel('Mean of $\mathbf{\hat{\omega}} (t)$')
ax.set_xlabel('Data streams index')
plt.savefig('out-of-control.png',bbox_inches='tight',dpi  = 500)

        
