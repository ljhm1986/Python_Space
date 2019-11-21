import pandas as pd
import numpy as np 
### 불러오기 ###
metascore = pd.read_csv("c:/WorkSpace/PythonSpace/Python_Space/Project/metascoreDF_NAME.csv")
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

#metacritic을 기준으로 본 년도별 출시된 게임의 수 
year_count = collections.Counter(metascore['year'])
type(year_count)
year_count.keys()
year_count.values()
plt.bar(year_count.keys(), year_count.values())
plt.xlabel('년도')
plt.ylabel('발매된 게임의 갯수')
plt.title('metacritic에 의한 년도별 발매 게임 수')
plt.show()

#년도별 점수 비교 
metascore.groupby('year')[['metascore','userscore']].mean()
metascore.groupby('year')[['metascore','userscore']].describe()

plt.scatter(metascore['metascore'],metascore['userscore'],s = 3)
plt.xlabel('metascore')
plt.ylabel('userscore')
plt.title('metascore와 userscore의 산점도')
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

Y = metascore['userscore']
LR.fit(X_2,Y)
LR1 = LR.fit(X_2,Y)
print(LR1)
print('기울기 : ',LR1.coef_)
print('절편 : ',LR1.intercept_)
coef1 = LR1.coef_[0]
intercept1 = LR1.intercept_

plt.scatter(X,Y,s = 3)
plt.plot(X,X*coef1 + intercept1, c = 'red')
plt.xlabel('metascore')
plt.ylabel('userscore')
plt.title('metacritic에서 metascore와 userscore의 관계')
plt.show()

#공분산
metascore['metascore'].cov(metascore['userscore'])
#상관계수 
metascore['metascore'].corr(metascore['userscore'])

def cal_mean(x):
    mean = 0
    for i in range(len(x)):
        mean += x[i]
    mean = mean / len(x)
    return mean

def mean_sq(x,y):

    if len(x) != len(y):
        print("not equal length")
        return None

    x_mean = cal_mean(x)
    y_mean = cal_mean(y)
    
    sum = 0
    for i in range(len(x)):
        sum += (x[i] - x_mean) * (y[i] - y_mean)

    return sum

def estimateLine(x,y):
    
    x_mean = cal_mean(x)
    print('X의 평균 : {}'.format(x_mean))

    y_mean = cal_mean(y)
    print('Y의 평균 : {}'.format(y_mean))

    Sxx = mean_sq(x,x)
    
    Syy = mean_sq(y,y)
    
    Sxy = mean_sq(x,y)
    
    #slope    
    b1 = Sxy / Sxx
    #intercept
    b0 = y_mean - (b1*x_mean)
    
    print("slope(b1) : {}, intercept(b0) : {}".format(b1,b0))

    SSE = 0
    SSR = 0
    for i in range(len(x)):
        y_esti = b0 + b1 * x[i]
        SSE += (y[i] - y_esti)**2
        SSR += (y_esti - y_mean)**2
    
    #sample coefficient of determination
    R2 = SSR/(SSR + SSE)
    print('표본결정계수 : {}'.format(R2))

    #standard error of estimate
    s_y_x = 0
    s_y_x = (SSE/ (len(x) - 2))**(1/2)
    print("추정값의 표준오차 : {}".format(s_y_x))

    #sample coefficient of correlation
    Rxy = Sxy / (Sxx * Syy)**(1/2)
    print("표본상관계수 : {}".format(Rxy))

    MSR = SSR
    MSE = SSE / (len(x) - 2)
    F0 = MSR / MSE
    print('F0 : {}'.format(F0))

    #H0 : b1 = 0 으로 할때의 t0 
    t0 = b1 / ((MSE/Sxx)**(1/2))
    print('t0 (H0 : b1 = 0) : {}'.format(t0))
    

estimateLine(X,Y)
""" 
X의 평균 : 70.8902405330733
Y의 평균 : 6.704389041118512
slope : 0.06814402121296771, intercept : 1.873642986440382
표본결정계수 : 0.33781328100776914
추정값의 표준오차 : 1.0647427435516357
표본상관계수 : 0.5812170687512254
F0 : 2091.6071138359744
t0 (H0 : b1 = 0) : 45.73409137433373 """

from scipy import stats
slope, intercept, r_value, p_value, stderr = stats.linregress(X, Y)
print(slope, intercept,r_value,p_value,stderr)


## metascore와 openscore의 관계 ##
LR = LinearRegression()
mergeDF = pd.read_csv("C:/WorkSpace/PythonSpace/Python_Space/Project/merge.csv")
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

plt.scatter(X,Y,s = 3)
plt.plot(X,X*coef2 + intercept2, c = 'red')
plt.xlabel('metascore')
plt.ylabel('openscore')
plt.title(' metascore와 openscore의 관계')
plt.show()

#공분산
X.cov(Y)
#상관계수
X.corr(Y)

estimateLine(X,Y)
"""
X의 평균 : 71.212017380509
Y의 평균 : 71.5243947858473
slope : 0.971079353183772, intercept : 2.371875009071104
표본결정계수 : 0.9212120607895761
추정값의 표준오차 : 3.1201877194989835
표본상관계수 : 0.9597979270604672
F0 : 31370.435422829636 
t0 (H0 : b1 = 0) : 177.11701054057363 """

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

estimateLine(X,Y)

from scipy import stats
slope, intercept, r_value, p_value, stderr = stats.linregress(X, Y)
print(slope, intercept,r_value,p_value,stderr)

#goty = 0,1,2 밖에 없으므로 일반적인 다변량 회귀가 별로이다.....
###장르도 넣어서 다변량 회귀를 하자 
mergeA
mergeB

MLR = LinearRegression()
X1 = mergeA.iloc[:,2:-1]
X1
Y = mergeA['goty']
MLR1 = MLR.fit(X1,Y)
MLR1.coef_
MLR1.intercept_

df1 = DataFrame({'factor':X1.columns,'slope':MLR1.coef_})
df1
df1['slope'].abs().sort_values()
df1['slope_abs'] = df['slope'].abs()
df1
df1.sort_values(by = 'slope_abs', ascending = False)

###장르만 넣어서 다변량 회귀를 하자 
mergeA
mergeB

MLR = LinearRegression()
X2 = mergeA.iloc[:,5:-1]
X2
Y = mergeA['goty']
MLR2 = MLR.fit(X2,Y)
MLR2.coef_
MLR2.intercept_

df2 = DataFrame({'factor':X2.columns,'slope':MLR2.coef_})
df2
df2['slope'].abs().sort_values()
df2['slope_abs'] = df2['slope'].abs()
df2
df2.sort_values(by = 'slope_abs', ascending = False)

## score 들과 genre들의 관계
