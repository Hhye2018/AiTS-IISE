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
import os
import pandas as pd
import time



os.chdir('/Users/hye42/OneDrive - UW-Madison/Quantile_antirank/Case study') #office computer

## Import standardized normal (in-control) and abnormal (out-of-control) data

df_norm = pd.read_csv('df_norm_standardized.csv')
df_abnorm = pd.read_csv('df_abnorm_standardized.csv')

df_norm.reset_index(drop=True, inplace=True)
df_norm1 = df_norm.values
df_norm1 = df_norm1.transpose()


def learn_rank(X):
    
    '''
    Function to learn the in-control probabilities based on historical in-controal data
    
    Input: a p by size matrix, where p is the number of data streams and size is number of observations
    
    Output: two p by 1 vectors (prob_min is for detecting downward mean shift, prob_max is for detecting upward mean shift)
    The sum of each vector equals to 1.
    
    '''
    
    p = X.shape[0]
    size = X.shape[1]
         
    rank_min = np.zeros([p,size])
    rank_max = np.zeros([p,size])

    for i in range(size):
        rank_min[np.argmin(X[:,i]),i] = 1
        rank_max[np.argmax(X[:,i]),i] = 1
    
    prob_min = np.sum(rank_min,axis = 1)/size
    prob_max = np.sum(rank_max,axis = 1)/size
    
    min_index = np.where(prob_min == 0)[0]
    max_index = np.where(prob_max == 0)[0]
    
    prob_min[min_index] = 0.00001
    prob_max[max_index] = 0.00001
    
    prob_min = prob_min/sum(prob_min)
    prob_max = prob_max/sum(prob_max)
    return [prob_min,prob_max]

def dirichlet_sample(data):
    
    '''
    Function to generate a random sample from dirichlet distribution
    '''
    rnd_dat = np.random.gamma(data,1)
    
    return rnd_dat/sum(rnd_dat)
    

def AiTS(ic,r,k,h,g_min, g_max):
    
    '''
    Main function of the AiTS algorithm
    
    Input: 
    ic: indicator to indicate whether the setup is for in-control or out-of-control
    r: the number of observable data streams ( r < p )
    k: the allowance parameter to restart the CUSUM (start choosing k = 0.1 if not sure)
    h: the prespecified threshold to raise the alarm
    g_min: the prob_min for detecting downward mean shift
    g_max: the prob_max for detecting upward mean shift
    
    Output:
    the run length, i.e. when the chart raises the alarm
    
    '''
    if ic == 1:
        randomlist = np.random.choice(1294, 3000)
        X = df_norm.iloc[randomlist,:]
        X.reset_index(drop=True, inplace=True)
        #X = X.drop(labels='classification', axis=1)
        X = X.values
        X= X.transpose()
    else:
        randomlist = np.random.choice(1294, 25)
        X0 = df_norm.iloc[randomlist,:]
        X0.reset_index(drop=True, inplace=True)
        #X = X.drop(labels='classification', axis=1)
        X0 = X0.values
        X0 = X0.transpose()
        
        randomlist_ab = np.random.choice(99,975)
        X1 = df_abnorm.iloc[randomlist_ab,:]
        X1.reset_index(drop=True, inplace=True)
        #X = X.drop(labels='classification', axis=1)
        X1 = X1.values
        X1 = X1.transpose()
        #X[0:5,25:1000] = X[0:5,25:1000] + 1
        
        X = np.concatenate((X0, X1), axis = 1)
    
    #time_start = time.perf_counter()
    p = np.shape(X)[0] ## the number of data streams in total
    q = np.shape(X)[1] ## the time epoch for monitoring
    
    fulllist = np.arange(0,p,1) 
    observed_up = np.arange(0,r,1)
    unobserved_up = np.setdiff1d(fulllist, observed_up)
    
    observed_down = np.arange(0,r,1)
    unobserved_down = np.setdiff1d(fulllist, observed_down)
    
    S1_down = np.zeros([p,q])
    S1_up = np.zeros([p,q])
    
    S2_down = np.zeros([p,q])
    S2_up = np.zeros([p,q])

    C_down  = np.zeros(q)
    C_up  = np.zeros(q)
  
    xi_down = np.zeros([p,q])    ## first antirank indicator for downward detection
    xi_up = np.zeros([p,q])      ## first rank indicator for upward detection
    
    theta_down = np.zeros([p,q]) ## estimate of categorical distribution for downward detection
    theta_up = np.zeros([p,q])   ## estimate of categorical distribution for upward detection

    y_down  = np.zeros(q) ## monitoring statistic for downward detection
    y_up  = np.zeros(q)   ## monitoring statistic for upward detection
    
    alpha_down = g_min ## parameter for dirichlet distribution for downward detection
    alpha_up = g_max   ## parameter for dirichlet distribution for upward detection
    
    y = np.zeros(q)

    i = 0
    
    weight_down = np.zeros([p])
    weight_up = np.zeros([p])
    
    X_data_down = np.copy(X[:,i])
    X_data_down[unobserved_down] = 10000
    
    X_data_up = np.copy(X[:,i])
    X_data_up[unobserved_up] = -10000

    
    weight_down[np.argmin(X_data_down)] = sum(g_min[observed_down])
    weight_up[np.argmax(X_data_up)] = sum(g_max[observed_up])
   
    weight_down[unobserved_down] = g_min[unobserved_down]
    weight_up[unobserved_up] = g_max[unobserved_up]
    
    alpha_down = alpha_down + weight_down 
    alpha_up = alpha_up + weight_up
    
    theta_down[:,i] = alpha_down/sum(alpha_down)
    theta_up[:,i] = alpha_up/sum(alpha_up)

    xi_down[np.argmin(X_data_down),i] = sum(theta_down[observed_down,i])
    xi_up[np.argmax(X_data_up),i] = sum(theta_up[observed_up,i])
    
    xi_down[unobserved_down,i] = theta_down[unobserved_down,i]
    xi_up[unobserved_up,i] = theta_up[unobserved_up,i]

    
    C_down[0] = np.matmul(np.matmul((xi_down[:,0]).transpose(), np.diag(1/g_min)), xi_down[:,0] )
    C_up[0] = np.matmul(np.matmul((xi_up[:,0]).transpose(), np.diag(1/g_max)), xi_up[:,0] )

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
   
    y_down[0] = max(0, C_down[0] - k)
    
    if C_up[0] <= k:
        S1_up[:,0] = S1_up[:,0]
        S2_up[:,0] = S2_up[:,0]
        observed_up = observed_up
        unobserved_up = np.setdiff1d(fulllist, observed_up)

    else:
        S1_up[:,0] = xi_up[:,0] * (1- k/C_up[0]) 
        S2_up[:,0] = g_max * (1-k/C_up[0])
        N = dirichlet_sample(S1_up[:,0]/sum(S1_up[:,0]))
        observed_up =   N.argsort()[::-1][:r]
        unobserved_up = np.setdiff1d(fulllist, observed_up)
    
    y_up[0] = max(0, C_up[0] - k)      
    
    y[0] = np.max([y_down[0],y_up[0]])
    
    i = 1
    
    while   (y[i-1] < h and i <= q - 1):
       
        weight_down = np.zeros([p])
        weight_up = np.zeros([p])
        
        X_data_down = np.copy(X[:,i])
        X_data_down[unobserved_down] = 10000
    
        X_data_up = np.copy(X[:,i])
        X_data_up[unobserved_up] = -10000
        
        weight_down[np.argmin(X_data_down)] = sum(g_min[observed_down])
        weight_up[np.argmax(X_data_up)] = sum(g_max[observed_up])
  
        weight_down[unobserved_down] = g_min[unobserved_down]
        weight_up[unobserved_up] = g_max[unobserved_up]
               
        alpha_down = alpha_down + weight_down 
        alpha_up = alpha_up + weight_up
        
        theta_down[:,i] = alpha_down/sum(alpha_down)
        theta_up[:,i] = alpha_up/sum(alpha_up)

        xi_down[np.argmin(X_data_down),i] = sum(theta_down[observed_down,i])
        xi_up[np.argmax(X_data_up),i] = sum(theta_up[observed_up,i])
        
        xi_down[unobserved_down,i] = theta_down[unobserved_down,i]
        xi_up[unobserved_up,i] = theta_up[unobserved_up,i]

    
        C_down[i] = np.matmul(np.matmul((S1_down[:,i-1]-S2_down[:,i-1]+xi_down[:,i]-g_min).transpose(), np.diag(1/(S2_down[:,i-1]+g_min))), (S1_down[:,i-1]-S2_down[:,i-1]+xi_down[:,i]-g_min) )
        C_up[i] = np.matmul(np.matmul((S1_up[:,i-1]-S2_up[:,i-1]+xi_up[:,i]-g_max).transpose(), np.diag(1/(S2_up[:,i-1]+g_max))), (S1_up[:,i-1]-S2_up[:,i-1]+xi_up[:,i]-g_max) )

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
           
        y_down[i] = max(0, C_down[i] - k)
        
        if C_up[i] <= k:
            S1_up[:,i] = S1_up[:,i]
            S2_up[:,i] = S2_up[:,i]
            observed_up = observed_up
            unobserved_up = np.setdiff1d(fulllist, observed_up)
        else:
            S1_up[:,i] = (S1_up[:,i-1] + xi_up[:,i]) * (1- k/C_up[i]) 
            S2_up[:,i] = (S2_up[:,i-1] + g_max ) * (1-k/C_up[i])
            N = dirichlet_sample(S1_up[:,i]/sum(S1_up[:,i]))
            observed_up =   N.argsort()[::-1][:r]
            unobserved_up = np.setdiff1d(fulllist, observed_up)
    
        y_up[i] = max(0, C_up[i] - k) 
        y[i]  = np.max([y_down[i], y_up[i]])
        
        i = i + 1
    #print((time.perf_counter() - time_start)/3)
    return(i - 1)

def learn(r,k,h,g_min,g_max):
 
    '''
    Parallel computation for average run lenght based on 5000 simulation runs when the process is in control
    '''
    
    T = 5000      
    num_cores = 25 
    
    Y = Parallel(n_jobs = num_cores,max_nbytes= None)(delayed(AiTS)(1,r,k,h,g_min, g_max) for i in range(T)) ## in-control set up 

    Y1 = np.asarray(Y)
    ARL = np.mean(Y1)
    std_ARL = np.std(Y1)/np.sqrt(len(Y1))  
    
    return [ARL, std_ARL] 



def learn_oc(r,k,h,g_min,g_max):

    '''
    Parallel computation for average run lenght based on 5000 simulation runs when the process is out of control
    '''
    
    T = 5000      
    num_cores = 25 
    
    Y = Parallel(n_jobs = num_cores,max_nbytes= None)(delayed(AiTS)(0,r,k,h,g_min, g_max) for i in range(T))
    Y1 = np.asarray(Y)
    return Y1 

def learnh_ic(r,k,g_min, g_max):

    '''
    Function to learn the threshold h based on in-control average run length, which is 370 here. 
    We use the bisection method to learn this threshold
    '''
    
    RL = 370
    small_delta = 2
    h1 = 10
    if r == 10:
        h2 = 20
    elif r == 30:
        h2 = 40
    elif r == 40:
        h2 = 50
    
    A,B = learn(r,k,h2,g_min,g_max)

    if abs(A-RL) < small_delta:
        print('it is finished')
    

    while A< RL:
        h2 = h2 + 3
        A,B = learn(r,k,h2,g_min,g_max)
        if abs(A- RL) < small_delta:
            print('it is finished')
    
    h1= h2/2
    h = (h1+h2)/2

    A,B = learn(r,k,h,g_min,g_max)

    if abs(A-RL) < small_delta:
        print('it is finished')

    while abs(A-RL)> small_delta and abs(h1-h2) > 0.00001:
        if A < RL:
            h1 = h
            h = (h1+h2)/2
            A,B = learn(r,k,h,g_min,g_max)
        else:
            h2 = h
            h = (h1+h2)/2
            A,B = learn(r,k,h,g_min,g_max)

    print(h)
    print('ARL0 is ' + str(A))

    return [h,A,B]

np.random.seed(1000000)
random.seed(1000000)


k = 0.1

[g_min, g_max] = learn_rank(df_norm1)

for r in [10]:
    YY = learnh_ic(r,k,g_min, g_max)

    ic_threshold = YY[0]
    
    output = learn_oc(r,k,ic_threshold,g_min,g_max)
    
    updated_oc = output[output >= 25] - 25 + 1
    
    print(str(np.mean(updated_oc)))
    
    if r == 10:
        np.savetxt('AiTS_st25_incontrol_r10.txt',YY, delimiter = ',')
        np.savetxt('AiTS_st25_out-of-control_r10.txt',updated_oc, delimiter = ',')
    elif r == 30:
        np.savetxt('AiTS_st25_incontrol_r30.txt',YY, delimiter = ',')
        np.savetxt('AiTS_st25_out-of-control_r30.txt',updated_oc, delimiter = ',')
    elif r == 40:
        np.savetxt('AiTS_st25_incontrol_r40.txt',YY, delimiter = ',')
        np.savetxt('AiTS_st25_out-of-control_r40.txt',updated_oc, delimiter = ',')
