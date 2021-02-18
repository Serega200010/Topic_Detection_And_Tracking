# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 09:48:47 2021

@author: Sergei
"""
"""Избавиться от for"""

from scipy.stats import entropy
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import re                    
import numpy as np
from numba import guvectorize
from ewm import ewma

@guvectorize(['void(float64[:], intp[:], float64[:])'],
             '(n),()->(n)')
def move_mean(a, window_arr, out):
    window_width = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count

#@guvectorize(['void(float64[:], intp[:], float64[:])'],
 #            '(n),()->(n)')
def move_weighted_mean_linear(a, window_arr, out):
    window_width = window_arr[0]
    weights = (1/2)**np.arange(window_width)
    vec = a[-window_width:]
    return np.dot(vec,weights)

        

        #Cosine Measures
def Cosine_measure(x,y):
    cos = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return cos
    
#The mean of cosine distances between news in chain and argument new
def Cosine_measure_to_chain_av(chain, new):
    Sum = np.array([cosine(np.array(c), np.array(new)) for c in chain ]).sum()
    Sum/= len(chain)
    #print(Sum)
    return Sum

#The weighted mean of cosine distances between news in chain and argument new
def Cosine_measure_to_chain_w_av(chain, new):
    distances = [cosine(np.array(c), new) for c in chain[-4:]]
    m = ewma(np.array(distances))
    return m[-1]
    
#The EWM of cosine distances between news in chain and argument new
def Cosine_measure_to_chain_asMin(chain, new):
        distances = np.array([cosine(np.array(c), new) for c in chain ])
        ans = distances.max()
        return ans
#The distance from new and chain as cosine measure between new and the last 
def Cosine_measure_to_chain_last(chain, new):
    distance = cosine(chain[-1], new)
    return distance


                            #Euclidian Measure
    
#The mean of Euclidian distances between news in chain and argument new
def Euclidian_measure_to_chain_av(chain, new):
    chain = np.array(chain)
    new= np.array(new)
    chain = chain.reshape(chain.shape[0],-1)
    arr = np.linalg.norm(chain - new,axis = 1)
    #arr = np.array([np.linalg.norm(np.array(c) - new) for c in chain ])
    h = move_mean(arr, 5)
   # print(len(h))
    return h[-1]

#The weighted mean of cosine distances between news in chain and argument new
def Euclidian_measure_to_chain_w_av(chain, new):
    distances = pd.Series([np.linalg.norm(np.array(c) - new) for c in chain ])
    mean = distances.ewm(3, min_periods = 0).mean()
    return mean.to_numpy()[-1]
    
#The EWM of cosine distances between news in chain and argument new
def Euclidian_measure_to_chain_asMin(chain, new):
        distances = np.array([np.linalg.norm(np.array(c) - new) for c in chain ])
        ans = distances.max()
        return ans
#The distance from new and chain as cosine measure between new and the last 
def Euclidian_measure_to_chain_last(chain, new):
    distance = np.linalg.norm(chain[-1] - new)
    return distance

                            #Manhattan Measure
#The mean of Euclidian distances between news in chain and argument new
def Manhattan_measure_to_chain_av(chain, new):
    chain = np.array(chain)
    new= np.array(new)
    chain = chain.reshape(chain.shape[0],-1)
    arr = np.sum(np.abs(chain - new),axis = 1)
    #arr = np.array([np.linalg.norm(np.array(c) - new) for c in chain ])
    h = move_mean(arr, 5)
    print(h)
    return h[-1]


#The weighted mean of cosine distances between news in chain and argument new
def Manhattan_measure_to_chain_w_av(chain, new):
    chain = np.array(chain)
    new= np.array(new)
    chain = chain.reshape(chain.shape[0],-1)
    arr = np.sum(np.abs(chain - new),axis = 1)
    #arr = np.array([np.linalg.norm(np.array(c) - new) for c in chain ])
    h = move_weighted_mean_linear(arr, 5)
    print(h)
    return h[-1]
    
    
    '''
    distances = pd.Series([np.sum(np.abs(np.array(c) - new)) for c in chain ])
    mean = distances.ewm(3, min_periods = 0).mean()
    return mean.to_numpy()[-1]'''
    
#The EWM of cosine distances between news in chain and argument new
def Manhattan_measure_to_chain_asMin(chain, new):
        distances = np.array([np.sum(np.abs(np.array(c) - new)) for c in chain ])
        ans = distances.max()
        return ans
#The distance from new and chain as cosine measure between new and the last 
def Manhattan_measure_to_chain_last(chain, new):
    distance = np.sum(np.abs(np.array(chain[-1]) - new))
    return distance



    
                            #KL-Divergence 
#The mean of KL-Divergence  between news in chain and argument new
def KL_measure_to_chain_av(chain, new):
    Sum = np.array([entropy(np.array(c), new) for c in chain ]).sum()
    Sum/= len(chain)
    return Sum

#The weighted mean of KL-Divergence  between news in chain and argument new
def KL_measure_to_chain_w_av(chain, new):
    distances = pd.Series([entropy(np.array(c) ,new) for c in chain ])
    mean = distances.ewm(3, min_periods = 0).mean()
    return mean.to_numpy()[-1]
    
#The EWM of KL-Divergence  between news in chain and argument new
def KL_measure_to_chain_asMin(chain, new):
        distances = np.array([entropy(np.array(c), new) for c in chain ])
        ans = distances.max()
        return ans
#The distance from new and chain as KL-Divergence  between new and the last 
def KL_measure_to_chain_last(chain, new):
    distance = entropy(chain[-1],new)
    return distance