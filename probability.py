# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:04:06 2019

@author: stu12
"""

import math
import statistics

x = 1
def normal(x):
    value = (1 / math.sqrt(2*math.pi) ) * math.e **(-x**2/2)
    return value

def integral_normal(start, end, l):
    i = start
    S = 0
    while(i <= end):
        S += (1/l) * normal(i)
        i += 1/l
    
    return S

integral_normal(0,2.58,100000)

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

plt.plot(x = range(-1,1), y = [integral_normal(i) for i in x])

statistics.mean([13, 18, 1, 10, 15 ,15 ,10 ,5, 23 ,20, 9, 11])
statistics.stdev([13, 18, 1, 10, 15 ,15 ,10 ,5, 23 ,20, 9, 11])
