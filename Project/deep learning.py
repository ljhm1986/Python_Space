# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:48:21 2019

@author: stu12
"""
#딥러닝(Deep Neural Network)
##11/18##

#AND 
#X1  	X2  	Y
#----	----	----
#0	    0	    0
#1	    0	    0
#0	    1	    0
#1    	1	    1

#w1 = 0.5, w2 = 0.5 , theta = =0.7
w1 = 0.5
w2 = 0.5
theta = 0.7
def AND(x1,x2):
    result = w1 * x1 + w2 * x2
    if result > theta:
        return 1
    else:
        return 0
    
AND(0,0)
AND(1,0)
AND(0,1)
AND(1,1)
# 실제 W1, W2 theta 값을 찾아서 해야 한다.

import numpy as np
inputData = np.array([[0,0],[0,1],[1,0],[1,1]])
inputData.shape
inputData[0][1]

print('AND Perceptron')
"""
[0,0] => 0
[0,1] => 0
[1,0] => 0
[1,1] => 1 """

def AND2(array):
    for i in array:
        temp = AND(i[0], i[1])
        print("[{}, {}] => {}".format(i[0],i[1],temp))
            
AND2(inputData)

for i in inputData:
    print(str(i) + " => " + str(AND(i[0],i[1])))
    
#np의 sum 을 이용하려면?
w = np.array([w1, w2])

def AND3(x1, x2):
    x = np.array([x1, x2])
    temp = np.sum(w*x)
    if  temp > theta:
        return 1
    else:
        return 0
   
AND3(1,1)     
"""
OR
X1	    X2      Y
----	----	----
0	    0	    0
1	    0	    1
0	    1	    1
1	    1	    1 """

def OR(x1,x2):
    x = np.array([x1, x2])
    temp = np.sum(w * x)
    
    if temp > 0:
        return 1
    else:
        return 0
    
for i in inputData:
    print(str(i) + " => " + str(OR(i[0],i[1])))
    
#NAND(NOT AND)
#X1	    X2	    Y
#----	----	----
#0	    0	    1
#1	    0	    1
#0	    1	    1
#1	    1	    0

def NAND(x1, x2):
    x = np.array([x1, x2])
        
    temp = np.sum(w * x)
        
    if temp > theta:
        return 0
    else: 
        return 1

for i in inputData:
    print(str(i) + " => " + str(NAND(i[0],i[1])))
    
#XOR(eXclusive OR)
#X1	    X2	    Y
#----	----	----
#0	    0	    0
#1	    0	    1
#0    	1	    1
#1	    1	    0
#XOR(NOT OR)는 단일 회로로는 안 된다.
#다층퍼셉트론(Multi layer Perceptron)
#X1	    X2	    OR	    NAND	AND(OR와 NAND의)
#----	----	----	----	----
#0	    0	    0	    1	    0
#1	    0	    1	    1	    1
#0	    1	    1	    1	    1
#1	    1	    1	    0	    0

def XOR(x1, x2): #0층
    s1 = OR(x1, x2) #1층
    s2 = NAND(x1, x2)#1층
    result = AND(s1,s2)#2층
    
    return result

for i in inputData:
    print(str(i) + " => " + str(XOR(i[0],i[1])))
    
#지금까지는 
w1 = 0.5
w2 = 0.5
b = -0.5

def AND4(x1, x2):
    w = np.array([w1, w2, b])
    x = np.array([x1, x2, 1])
    temp = np.sum(w*x)
    
    if temp > 0:
        return 1
    else:
        return 0      
    
for i in inputData:
    print(str(i) + " => " + str(AND4(i[0],i[1])))

b = -0.2
def OR2(x1, x2):
    w = np.array([w1, w2, b])
    x = np.array([x1, x2, 1])
    temp = np.sum(w*x)
    
    if temp > 0:
        return 1
    else:
        return 0 
    
for i in inputData:
    print(str(i) + " => " + str(OR2(i[0],i[1])))
    
def NAND2(x1, x2):
    w1, w2, b = -0.5, -0.5, 0.7
    w = np.array([w1, w2, b])
    x = np.array([x1, x2, 1])
        
    temp = np.sum(w * x)
        
    if temp > 0:
        return 1
    else: 
        return 0

for i in inputData:
    print(str(i) + " => " + str(NAND2(i[0],i[1])))
    

#step ft의 함수를 만들자 
def step_function(x):
    
    if x > 0:
        return 1
    else:
        return 0
    
step_function(4)
step_function(10)
step_function(-2)

def step_function2(array):
    
    for i in array:
        if i > 0:
            print(1)
        else:
            print(0)
            
step_function2(np.array([1]))
step_function2(np.array([1,-1,10]))

[1 if i > 0 else 0 for i in np.array([1,-1,10])]

x = np.array([1,10,-1])
y = x > 0
y
y.astype(np.int)

#bool -> int변환 : True -> 1, False -> 0

def step_ft3(array):
    y = x > 0
    return y.astype(np.int)

step_ft3(x)

#step ft의 그래프를 그려보자
import matplotlib.pylab as plt
x = np.arange(-5.0,5.0,0.1)
x
y = step_ft3(x)
y
plt.plot(x,y)

#sigmoid ft의 그래프를 그려보자
x = np.arange(-10.0,10.0,0.1)
x
y = 1 / (1 + np.exp(-x))
y
plt.plot(x,y)

def sigmoid(x):
    
    return 1 / (1 + np.exp(-x))

sigmoid(x)
