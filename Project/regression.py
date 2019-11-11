import pandas as pd
import numpy as np 
### 불러오기 ###
metascoreDF = pd.read_csv("c:/WorkSpace/PythonSpace/Python_Space/Project/metascore.csv")
metascoreDF = metascoreDF.iloc[:,1:]

opencriticDF = pd.read_csv("c:/WorkSpace/PythonSpace/Python_Space/Project/opencritic.csv")
opencriticDF = opencriticDF.iloc[:,1:]
### 

metascoreDF.columns
metascoreDF.info()
opencriticDF.columns
opencriticDF.info()

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

plt.scatter(metascoreDF['metascore'],metascoreDF['userscore'],s = 5)
plt.show()

## metascore와 userscore의 관계 ## 
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
X = metascoreDF['metascore']
X.shape #(N,) -> (N,1) 이렇게 바꿔줘야 한다. 2차원배열 
#(N,) -> (1,N) -> (N,1) 이런 과정으로 할 것이다.
type(X)

X_1 = np.atleast_2d(X)
X_1.shape
type(X_1)
#Series가 ndarray로 변환되었다.

X_2 = np.transpose(X_1)
X_2.shape

Y = metascoreDF['userscore']*10
LR.fit(X_2,Y)
LR1 = LR.fit(X_2,Y)
print(LR1)
print('기울기 : ',LR1.coef_)
print('절편 : ',LR1.intercept_)
coef1 = LR1.coef_[0]
intercept1 = LR1.intercept_

plt.scatter(metascoreDF['metascore'],metascoreDF['userscore']*10,s = 5)
plt.plot(metascoreDF['metascore'],metascoreDF['metascore']*coef1 + intercept1,
        c = 'red')
plt.show()

#공분산
metascoreDF['metascore'].cov(metascoreDF['userscore'])
#상관계수 
metascoreDF['metascore'].corr(metascoreDF['userscore'])

## metascore와 openscore의 관계 ##
LR = LinearRegression()
mergeDF = pd.read_csv("C:/WorkSpace/Python_Space/Project/merge.csv")
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
