# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:40:08 2019

@author: STU24
"""

import numpy as np

y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
t = [0,0,1,0,0,0,0,0,0,0]

#평균제곱오차  
def mean_squared_error(y,t):
    y = np.array(y)
    t = np.array(t)
    
    return 0.5*np.sum((y-t) ** 2)

mean_squared_error(y,t)

y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
mean_squared_error(y,t)

y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
mean_squared_error(y,t)

#교차 엔트로피 오차
def cross_entropy_error(y,t):
    y = np.array(y)
    t = np.array(t)
    
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
t = [0,0,1,0,0,0,0,0,0,0]
cross_entropy_error(y,t)

y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
cross_entropy_error(y,t)



with open()