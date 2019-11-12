import pandas as pd
import numpy as np 
### 불러오기 ###
metascore = pd.read_csv("c:/WorkSpace/Python_Space/Project/metascoreDF_NAME.csv")
metascore = metascore.iloc[:,1:]

opencritic = pd.read_csv("c:/WorkSpace/Python_Space/Project/opencriticAll.csv")
opencritic = opencritic.iloc[:,1:]
### 

metascore.columns
metascore.info()
opencritic.columns
opencritic.info()

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

plt.scatter(metascore['metascore'],metascore['userscore'],s = 5)
plt.show()

## metascore와 userscore의 관계 ## 
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
X = metascore['metascore']
X.shape #(N,) -> (N,1) 이렇게 바꿔줘야 한다. 2차원배열 
#(N,) -> (1,N) -> (N,1) 이런 과정으로 할 것이다.
type(X)

X_1 = np.atleast_2d(X)
X_1.shape
type(X_1)
#Series가 ndarray로 변환되었다.

X_2 = np.transpose(X_1)
X_2.shape

Y = metascore['userscore']*10
LR.fit(X_2,Y)
LR1 = LR.fit(X_2,Y)
print(LR1)
print('기울기 : ',LR1.coef_)
print('절편 : ',LR1.intercept_)
coef1 = LR1.coef_[0]
intercept1 = LR1.intercept_

plt.scatter(X,Y,s = 5)
plt.plot(X,X*coef1 + intercept1, c = 'red')
plt.show()

#공분산
metascore['metascore'].cov(metascore['userscore'])
#상관계수 
metascore['metascore'].corr(metascore['userscore'])



def estimateLine(x,y):
    x_mean = 0
    for i in range(len(x)):
        x_mean += x[i]
    x_mean = x_mean / len(x)
    
    y_mean = 0
    for i in range(len(y)):
        y_mean += x[i]
    y_mean = y_mean / len(y)

    Sxx = 0
    for i in range(len(x)):
        Sxx += (x[i] - x_mean)**2
    Syy = 0
    for i in range(len(y)):
        Syy += (y[i] - y_mean)**2
    
    Sxy = 0
    for i in range(len(x)):
        Sxy += (x[i] - x_mean) * (y[i] - y_mean)
    
    #slope    
    b1 = Sxy / Sxx
    #intercept
    b0 = y_mean - (b1*x_mean)
    
    print("slope : {}, intercept : {}".format(b1,b0))

estimateLine(X,Y)

from scipy import stats
slope, intercept, r_value, p_value, stderr = stats.linregress(X, Y)
print(slope, intercept)
## metascore와 openscore의 관계 ##
LR = LinearRegression()
mergeDF = pd.read_csv("C:/WorkSpace/Python_Space/Project/merge.csv")
mergeDF = mergeDF.iloc[:,1:]
mergeDF.info()
X = mergeDF['metascore']
X.shape

X_1 = np.atleast_2d(X)
X_2 = np.transpose(X_1)
Y = mergeDF['openscore']
LR.fit(X_2,Y)
LR2 = LR.fit(X_2,Y)
print(LR2)
print('기울기 : ',LR2.coef_)
print('절편 : ',LR2.intercept_)
coef2 = LR2.coef_[0]
intercept2 = LR2.intercept_

plt.scatter(X,Y,s=5)
plt.plot(X,X*coef2+intercept2, c = 'red')
plt.show()

#공분산
X.cov(Y)
#상관계수
X.corr(Y)

#그냥 해본 year와 metascore의 관계 
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
X = metascore['metascore']
X.shape #(N,) -> (N,1) 이렇게 바꿔줘야 한다. 2차원배열 
#(N,) -> (1,N) -> (N,1) 이런 과정으로 할 것이다.
type(X)

X_1 = np.atleast_2d(X)
X_1.shape
type(X_1)
#Series가 ndarray로 변환되었다.

X_2 = np.transpose(X_1)
X_2.shape

Y = metascore['year']
LR.fit(X_2,Y)
LR3 = LR.fit(X_2,Y)
print(LR1)
print('기울기 : ',LR3.coef_)
print('절편 : ',LR3.intercept_)
coef3 = LR3.coef_[0]
intercept3 = LR3.intercept_

plt.scatter(X,Y,s = 5)
plt.plot(X,X*coef3 + intercept3, c = 'red')
plt.show()

#공분산
metascore['metascore'].cov(metascore['year'])
#상관계수 
metascore['metascore'].corr(metascore['year'])
