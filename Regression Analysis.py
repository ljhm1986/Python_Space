# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:29:44 2019

@author: stu12
"""

#10/31
#공분산을 구해보자 

x = [184,170,180]
y = [85,70,82]

import numpy as np

np.mean(x)
np.mean(y)
np.var(x)
np.var(y)
np.cov(x,y)

np.cov(x,y)[0][1]

#pandas를 이용해서 공분산을 구해보자 
import pandas as pd

pd.DataFrame(x)
pd.DataFrame(y)

df = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis = 1)
df.columns = ['x','y']
df['x'].cov(df['y'])

#상관계수를 구해보자 
np.corrcoef(x,y)[0][1]
df['x'].corr(df['y'])

#회귀선을 구해보자
from scipy import stats
height = [176,172,182,160,163,165,168,163,182,182]
weight = [72,70,70,43,48,54,51,52,73,88]

#독립변수를 앞에, 종속변수를 뒤에 넣는다.
slope, intercept, r_value, p_value, stderr = stats.linregress(height, weight)
#r_value : 상관계수 
#stderr : 추정값의 표준오차 

##
import csv
score_data =  []
with open("C:\\WorkSpace\\Python_Space\\data\\score.txt",
          'r', encoding = 'utf-8') as f:
    score = csv.reader(f)
    for i in score:
        score_data.append(i)
    
score_data
len(score_data)




#한글이 있으니 utf-8로 저장
score = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\score.txt")
score
slope, intercept, r_value, p_value, stderr = stats.linregress(
        score['IQ'], score['성적'])

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

plt.scatter(score['성적'],score['IQ'])

#독립변수를 여러개를 넣어보자 
score.iloc[:,2:]
slope, intercept, r_value, p_value, stderr = stats.linregress(
        score.iloc[:,2:], score['성적'])
#시행되지 않는다.

#다음의 라이브러리를 이용하자 
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(score.iloc[:,2:], score['성적'])

print("절편 : ", reg.intercept_)
print("기울기 : ",reg.coef_)

#책을 참고하면서
#challenger.csv와 insurance.csv을 읽어들이고 회귀선을 그려보자
challenger = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\challenger.csv")
insurance = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\insurance.csv")

insurance.columns

#더미 데이터를 만들기, 이전에 했던 방식은
insurance['gender_female'] = [1 if i == 'female' else 0 for i in insurance['gender']]
#를 하면 되는데, get_dummies()를 이용하자 
dumm_gender = pd.get_dummies(insurance['gender'], prefix = 'gender')
print(dumm_gender.head())
dumm_smoker = pd.get_dummies(insurance['smoker'], prefix = 'smoker')
print(dumm_smoker.head())

#두 df을 join하자 
insurance1 = dumm_gender.join(dumm_smoker)

insurance[['age','bmi','children','charges']]
#그럼 수치형데이터들만 조인하자 
insurance2 = insurance1.join(insurance[['age','bmi','children','charges']])
type(insurance2)
insurance2.info()

#이제 회귀모형에 학습하자 
reg = linear_model.LinearRegression()

col = ['age','bmi','children','gender_male','smoker_yes']
insurance2[col]

reg.fit(insurance2[col],insurance2['charges'])

print("절편 : ", reg.intercept_)
print("기울기 : ",reg.coef_)

#########################################################################
#11/1#
#알레르기 증세에 효과가 있다고 하는 새로 개발된 약품의 복용량(mg)과
#지속되는 기간(일)을 조사한 자료입니다.
#.....

#
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
iris = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\iris.csv")
iris
iris.info()
X = iris.iloc[:,:-1]
X
Y = iris['Name']

log = LogisticRegression()
log.fit(X,Y)

test1 = [[5.1, 3.5, 1.4, 0.2]]
log.predict(test1)

#실제 데이터를 넣어보자 
iris.iloc[7,0:4]
type(test1)
type([iris.iloc[7,:0:4]])
log.predict([iris.iloc[7,0:4]])
#array(['Iris-setosa'], dtype=object)
print(iris.iloc[7,4])
#Iris-setosa

