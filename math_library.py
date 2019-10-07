# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:28:39 2019

@author: stu11
"""

import math

math.cos(1)
math.sin(math.pi)# 0이 아니네 
math.pi
math.e
math.tau


math.sin(0)
math.cos(0)
math.cos(math.pi/2)# 0이 아니네 

math.sin(math.pi) == 0 #False !!!

math.nan
math.inf

math.sin()

math.radians(90)

from pandas import Series, DataFrame

def ft(x):
    return math.sin(math.radians(x))

x = list(range(1,100))
x
y = []
for i in x:
    y.append(ft(i))
    
y

df = DataFrame(x,y)
df

import matplotlib.pylab as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

plt.plot(x,y)
plt.show()

import numpy as np
np.pi
np.sin(np.pi)

(np.sin(np.pi) - math.sin(math.pi)) == 0 #True

