# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:05:27 2019

@author: stu12
"""

import pandas as pd

iris = pd.read_csv("C:\WorkSpace\Python_Space\data\iris.csv")

iris.info()

x = iris.drop('Name',axis = 1)
y = iris['Name']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

import collections
collections.Counter(iris['Name'])
#갯수가 잘 분배되어 있는지 확인하자 
collections.Counter(y_train)
collections.Counter(y_test)

from sklearn.tree import DecisionTreeClassifier
#entropy 기반 모델을 사용
model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)

#max_depth : 가장 깊은 노드 

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

model.score(x_train, y_train)
model.score(x_test, y_pred)

#새로운 데이터를 넣어서 어떻게 분류되는지 보자 
model.predict([[5.1, 3.5, 5.1, 4.0]])[0]

#아래 주소로 들어가서 파일을 받기 
#https://graphviz.gitlab.io/_pages/Download/Download_windows.html
#graphviz 프로그램 설치 

#C:\Program Files (x86)\Graphviz2.38\bin
#시스템속성-환경변수-path 추가 
#anaconda command line 에서 pip install pudotplus, pip install graphviz
#TypeError : 'module' object is not callable 이 뜨며 안되면
#python -m pip install pydotplus --user, python -m pip install graphviz --user
#입력 하기 
import pydotplus
import graphviz

from sklearn.tree import export_graphviz
from IPython.display import Image

dot_data = export_graphviz(model, out_file=None,
                           feature_names=['SepalLength','SepalWidth',
                                          'PetalLength','PetalWidth'],
                           class_names=['Iris-setosa','Iris-virginca',
                                        'Iris-virsicolor'],
                           filled=True,rounded=True,special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

#이번에는 gini 기반으로도 그리자 
model = DecisionTreeClassifier(criterion = 'gini', max_depth = 3)

##titanic.csv 
titanic = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\titanic.csv")
titanic.info()
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 891 entries, 0 to 890
#Data columns (total 11 columns):
#survived    891 non-null int64
#pclass      891 non-null int64
#name        891 non-null object
#gender      891 non-null object
#age         714 non-null float64
#sibsp       891 non-null int64
#parch       891 non-null int64
#ticket      891 non-null object
#fare        891 non-null float64
#cabin       204 non-null object
#embarked    889 non-null object
#dtypes: float64(2), int64(4), object(5)
#memory usage: 76.7+ KB
#survived 목표변수 (0 : 사망, 1 : 생존), 
#embarked : 탑승항구

#sklearn 은 수치형 데이터로 바꿔야 한다.
#설명변수 : sklearn에 입력할 설명변수들은 문자변수가 사용될 수 없고,
#숫자변수만 사용이 가능하다. 
#예로 gender를 보자 
collections.Counter(titanic['gender'])
#Counter({'male': 577, 'female': 314})
#gender는 문자로 되어 있으니 숫자형으로 바꿔야 한다.
#gender -> 0과 1로 변경작업이 필요하다.(one hot encoding)
#female -> 0, male -> 1 로 encoding해 보자 

#titanic['gender']
#
#titanic['gender'] = [0 if i == 'female' else 1 for i in titanic['gender']]
#titanic['gender']
#
#값을 빠르게 바꾸는 함수, map()을 사용하자 
titanic['gender'] = titanic.gender.map({'female' : 0, 'male' : 1})
titanic['gender']

#NA를 찾아보자 
titanic.isnull() 
#NA가 하나 이상 있으면 True, 설명변수 중에서 NA이 있는 항목 확인 
titanic.isnull().any()
titanic.isnull().sum()#NA의 갯수가 나온다.
titanic.isnull().sum()['age']
titanic.isnull().sum().sum()#총 NA의 갯수 

collections.Counter(titanic['age'].isnull())
collections.Counter(titanic['cabin'].isnull())
collections.Counter(titanic['embarked'].isnull())

#NA인거 다 지운다고 하면 데이터의 양이 부족해진다.
#age의 평균을 NA에 넣자
titanic['age'].mean()#29.7
#
titanic['age'].median()#28.0

#중앙값을 채워넣자
titanic['age'].fillna(titanic['age'].median(), inplace = True)
titanic[titanic['age'].isnull()]['age']

#
titanic['embarked']
collections.Counter(titanic['embarked'])
#Counter({'S': 644, 'C': 168, 'Q': 77, nan: 2})
titanic['embarked'].unique()
#S(Southampton), Q(Queenstown), C(Cherbourg)
#이것도 수치형으로 바꾸어야 하는데 
#본래 one hot encoding 

#그럼 새로운 열을 만들자, 더미 테이블 생성 
pd.get_dummies(titanic.embarked, prefix = 'embarked')
embarked_dummies = pd.get_dummies(titanic.embarked, prefix = 'embarked')
embarked_dummies.info()
#이중에서 embarked_C column을 지우자 
embarked_dummies.drop(embarked_dummies.columns[0], axis = 1, inplace = True)
embarked_dummies

#이제 embarked_dummies를 titanic 에 붙이자 
titanic = pd.concat([titanic, embarked_dummies], axis=1)
titanic.info()

titanic.columns

#설명변수
feature_cols = ['pclass','gender','age','embarked_Q','embarked_S']
x = titanic[feature_cols]
x.head()

#목표변수
y = titanic.survived
y
#decision tree model
titanic_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)

titanic_model.fit(x,y)

pd.DataFrame({'feature':feature_cols,
              'importance':titanic_model.feature_importances_})

titanic_model.score(x,y)
#약 82.04%, 그리 좋은 점수는 아니다.

score = []
for i in range(1,50):
    titanic_model = DecisionTreeClassifier(criterion = 'entropy',
                                           max_depth = i)
    titanic_model.fit(x,y)
    score.append((i,titanic_model.score(x,y)))

score
#max_depth = 14 일때부터 약 90.01%로 나온다.

#max_depth = 14 일때 
titanic_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 14)
titanic_model.fit(x,y)
titanic_model.score(x,y)
dot_data = export_graphviz(titanic_model,
                           out_file=None,
                           feature_names= feature_cols,
                           class_names=['0','1'],
                           filled=True,rounded=True,special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())# 어 그림이 크다....

