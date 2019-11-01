# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:56:42 2019

@author: stu12
"""

##Boston.csv를 읽어들이고 회귀분석과 결정트리를 해 보자
#데이터명 : Boston Housing Price (보스턴 주택 가격 데이터) 
#레코드수 : 506 개 
#필드개수 :  14 개 
#
#데이터설명 : 보스턴 시의 주택 가격에 대한 데이터이다. 
#주택의 여러가진 요건들과 주택의 가격 정보가 포함되어 있다. 
#주택의 가격에 영향을 미치는 요소를 분석하고자 하는 목적으로 사용될 수 있다. 
#회귀분석 등의 분석에 활용될 수 있다. 
#
#보스턴 주택 데이터는 여러 개의 측정지표들 (예를 들어, 범죄율, 학생/교사 비율 등)을
#포함한, 보스턴 인근의 주택 가격의 중앙값(median value)이다. 
#이 데이터 집합은 14개의 변수를 포함하고 있다. 
#
#필드의 이해 : 
#데이터의 이해를 돕기 위해 포함된 14개의 변수에 대하여 간략하게 설명한다.  
#
#
#
# [01]  CRIM 자치시(town) 별 1인당 범죄율 
# [02]  ZN 25,000 평방피트를 초과하는 거주지역의 비율 
# [03]  INDUS 비소매상업지역이 점유하고 있는 토지의 비율 
# [04]  CHAS 찰스강에 대한 더미변수(강의 경계에 위치한 경우는 1, 아니면 0) 
# [05]  NOX 10ppm 당 농축 일산화질소 
# [06]  RM 주택 1가구당 평균 방의 개수 
# [07]  AGE 1940년 이전에 건축된 소유주택의 비율 
# [08]  DIS 5개의 보스턴 직업센터까지의 접근성 지수 
# [09]  RAD 방사형 도로까지의 접근성 지수 
# [10]  TAX 10,000 달러 당 재산세율 
# [11]  PTRATIO 자치시(town)별 학생/교사 비율 
# [12]  B 1000(Bk-0.63)^2, 여기서 Bk는 자치시별 흑인의 비율을 말함. 
# [13]  LSTAT 모집단의 하위계층의 비율(%) 
# [14]  MEDV 본인 소유의 주택가격(중앙값) (단위: $1,000) 

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

boston = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\boston.csv")

boston.info()
#보니까 모두 수치형 데이터이다.
boston.head()
boston.columns

from sklearn import linear_model
x = boston.iloc[:,:-1]
y = boston['MEDV']


type(x)
type(y)
model = linear_model.LinearRegression()
model.fit(x,y)
model.

print("절편 : ", model.intercept_)
print("기울기 : ",model.coef_)

type(model)
type(model.intercept_)
type(model.coef_)

print(model)
model.get_params()


slope = model.coef_

df = DataFrame({'name' :boston.columns[:-1], 'slope':slope})
df.info()
df
#       name      slope
#0      CRIM  -0.108011
#1        ZN   0.046420
#2     INDUS   0.020559
#3      CHAS   2.686734
#4       NOX -17.766611
#5        RM   3.809865
#6       AGE   0.000692
#7       DIS  -1.475567
#8       RAD   0.306049
#9       TAX  -0.012335
#10  PTRATIO  -0.952747
#11        B   0.009312
#12    LSTAT  -0.524758

df['slope'].abs().sort_values()
df['slope_abs'] = df['slope'].abs()
df
df.sort_values(by = 'slope_abs', ascending = False)
#       name      slope  slope_abs
#4       NOX -17.766611  17.766611
#5        RM   3.809865   3.809865
#3      CHAS   2.686734   2.686734
#7       DIS  -1.475567   1.475567
#10  PTRATIO  -0.952747   0.952747
#12    LSTAT  -0.524758   0.524758
#8       RAD   0.306049   0.306049
#0      CRIM  -0.108011   0.108011
#1        ZN   0.046420   0.046420
#2     INDUS   0.020559   0.020559
#9       TAX  -0.012335   0.012335
#11        B   0.009312   0.009312
#6       AGE   0.000692   0.000692

model1 = linear_model.LinearRegression()
type(boston.iloc[:,0])
model1.fit(boston.iloc[:,0:2],y)
model1.intercept_
model1.coef_

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

plt.scatter(boston.iloc[:,0],boston.iloc[:,-1])
plt.xlabel('CRIM')
plt.ylabel('MEDV')

#y = a + b * x
def SimpleLinearReg(x,y):
    x_mean = x.mean()
    y_mean = y.mean()
    
    S_xx = 0
    for i in range(len(x)):
        S_xx = S_xx + (x[i] - x_mean)**2
    
    S_xy = 0
    for i in range(len(x)):
        S_xy = S_xy + (x[i] - x_mean) * (y[i] - y_mean)

    b = S_xy / S_xx
    a = y_mean - b * x_mean
    
    return a, b

boston['NOX'].mean()
boston.iloc[:,0].mean()
boston.iloc[:,0][1]


from scipy import stats
slope, intercept, r_value, p_value, stderr = stats.linregress(boston['NOX'],boston['MEDV'])
slope
intercept

def RegPlot(x,y):
    
    a,b = SimpleLinearReg(x,y)
    print("절편 : {}, 회귀계수 : {}".format(a,b))
    plt.scatter(x,y)
    plt.plot(x,a + b * x, c= 'red')
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.show()
    

#회귀계수의 절대값이 큰 독립변수와 종속변수들 간의 산점도 
a, b = SimpleLinearReg(boston['NOX'],boston['MEDV'])
x = boston['NOX']
x.name
plt.scatter(boston['NOX'],boston['MEDV'])
plt.plot(x,a + b * x, c = 'red')

RegPlot(boston['RM'],boston['MEDV'])
RegPlot(boston['CHAS'],boston['MEDV'])
RegPlot(boston['DIS'],boston['MEDV'])
RegPlot(boston['PTRATIO'],boston['MEDV'])
RegPlot(boston['LSTAT'],boston['MEDV'])
RegPlot(boston['CRIM'],boston['MEDV'])

##
pd.set_option("display.max_columns",20)
pd.set_option("display.width",1000)
boston.mean()
boston.describe()
#             CRIM          ZN       INDUS        CHAS         NOX          RM  \
#count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000   
#mean     3.613524   11.363636   11.136779    0.069170    0.554695    6.284634   
#std      8.601545   23.322453    6.860353    0.253994    0.115878    0.702617   
#min      0.006320    0.000000    0.460000    0.000000    0.385000    3.561000   
#25%      0.082045    0.000000    5.190000    0.000000    0.449000    5.885500   
#50%      0.256510    0.000000    9.690000    0.000000    0.538000    6.208500   
#75%      3.677082   12.500000   18.100000    0.000000    0.624000    6.623500   
#max     88.976200  100.000000   27.740000    1.000000    0.871000    8.780000   
#
#              AGE         DIS         RAD         TAX     PTRATIO           B  \
#count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000   
#mean    68.574901    3.795043    9.549407  408.237154   18.455534  356.674032   
#std     28.148861    2.105710    8.707259  168.537116    2.164946   91.294864   
#min      2.900000    1.129600    1.000000  187.000000   12.600000    0.320000   
#25%     45.025000    2.100175    4.000000  279.000000   17.400000  375.377500   
#50%     77.500000    3.207450    5.000000  330.000000   19.050000  391.440000   
#75%     94.075000    5.188425   24.000000  666.000000   20.200000  396.225000   
#max    100.000000   12.126500   24.000000  711.000000   22.000000  396.900000   
#
#            LSTAT        MEDV  
#count  506.000000  506.000000  
#mean    12.653063   22.532806  
#std      7.141062    9.197104  
#min      1.730000    5.000000  
#25%      6.950000   17.025000  
#50%     11.360000   21.200000  
#75%     16.955000   25.000000  
#max     37.970000   50.000000  
boston.corr().round(2)

## 새로운 라이브러리 
import seaborn as sns
m = boston.corr().round(2)
m
#그림이 그려진다. 
sns.heatmap(data = m, annot = True)


##
features = ['CRIM','LSTAT','RM','B']
target = boston['MEDV']

#여러개의 산점도를 한 번에 출력해보자 
for i, col in enumerate(features):
    plt.subplot(1, len(features),i+1)
    x = boston[col]
    y = target
    plt.scatter(x,y,marker = 'o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
    
stats.linregress(boston['RM'],boston['MEDV'])
slope, intercept, r_value, p_value, stderr = stats.linregress(
        boston['RM'],boston['MEDV'])

plt.scatter(boston['RM'],boston['MEDV'])
plt.plot(boston['RM'],boston['RM']*slope + intercept, c = 'red')

#단순회귀일때 LinearRegression()을 사용하는 방법
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
x = boston['RM']
x.shape #(506,) -> (506,1) 이렇게 바꿔줘야 한다. 2차원배열 
#(506,) -> (1,506) -> (506,1) 이런 과정으로 할 것이다.
type(x)

x_1 = np.atleast_2d(x)
x_1.shape
type(x_1)
#Series가 ndarray로 변환되었다.

x_2 = np.transpose(x_1)
x_2.shape

y = boston['MEDV']
lr.fit(x_2,y)
l = lr.fit(x_2,y)
print(l)
print('기울기 : ',l.coef_)
print('절편 : ',l.intercept_)
