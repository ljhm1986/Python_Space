# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:33:28 2019

@author: stu12
"""
import numpy as np

def cost(x,y):
    
    temp = y * np.log(x) + (1-y) * np.log(1 - x)
    
    return temp

x = np.array(range(1,10,1))*(1/10)
y = np.array(range(1,10,1))*(1/10)

xy = [[i,j] for i in x for j in y]
xy = np.array(xy)

xy[:,0]

X, Y = np.meshgrid(xy[:,0], xy[:,1])

z = cost(X,Y)
z

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

axes = Axes3D(fig)

axes.plot_surface(X,Y,z,rstride = 1, cstride = 1, cmap = cm.viridis)
plt.xlabel()
plt.show()
