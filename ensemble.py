# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:40:01 2019

@author: stu12
"""

#11/6
import pandas as pd

#전에 했던거 다시 해 보자 
iris = pd.read_csv("C:\WorkSpace\Python_Space\data\iris.csv")

iris.info()

iris['Name']

x = iris.iloc[:,:4]
y = iris.iloc[:,4]
x
y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y)
type(x_train)#DataFrame
type(y_train)#Series
y_train.value_counts()
y_test.value_counts()

from sklearn.tree import DecisionTreeClassifier

#max_depth 가 크면 overfitting 이 나올 가능성이 높다.
model1 = DecisionTreeClassifier(criterion= 'entropy', max_depth= 5)

model1.fit(x_train, y_train)
predict_result = model1.predict(x_test)

import collections

collections.Counter(predict_result == y_test)

model1.score(x_train, y_train)
model1.score(x_test, y_test)
model1.classes_

model1.predict([[5.1,3.5,1.4,0.2]])[0]

model2 = DecisionTreeClassifier(criterion = 'gini', max_depth = 5)

model2.fit(x_train, y_train)
predict_result2 = model2.predict(x_test)

collections.Counter(predict_result2 == y_test)

#ensemble 을 해 보자 
from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_estimators = 10,#10개 만들기
                               oob_score = True,#default = False
                               random_state = 0)
#oob : sample할때 남아있는 데이터, oob_score : oob로 test한 점수 

model.fit(x_train, y_train)
RF_result = model.predict(x_test)
model.score(x_train, y_train)
model.score(x_test, y_test)
model.classes_
model1.predict([[5.1,3.5,1.4,0.2]])[0]

#정확도 점수를 확인하자 
from sklearn.metrics import accuracy_score

accuracy_score(y_test, RF_result)

#독립변수의 중요도를 확인해보자 
f = model.feature_importances_
r = pd.Series(f,index = x_train.columns)
#SepalLength    0.136724
#SepalWidth     0.040115
#PetalLength    0.325855
#PetalWidth     0.497306
#dtype: float64
s = r.sort_values(ascending = False)
#PetalWidth     0.497306
#PetalLength    0.325855
#SepalLength    0.136724
#SepalWidth     0.040115
#dtype: float64

import matplotlib.pyplot as plt
import seaborn as sns

plt.Figure(figsize=(10,10))
plt.title("feature importance")
sns.barplot(x = s, y = s.index)

##
#fixed.acidity       : 고정 산도
#volatile.acidity    : 휘발성 산도
#citric.acid         : 시트르산
#residual.sugar      : 잔류 설탕
#chlorides           : 염화물
#free.sulfur.dioxide : 자유 이산화황
#total.sulfur.dioxide: 총 이산화황
#density             : 밀도
#pH                  : pH
#sulphates           : 황산염
#alcohol             : 알코올
#quality             : 품질
whitewines = pd.read_csv("C:\WorkSpace\Python_Space\data\whitewines.csv")
whitewines.info()
whitewines

x = whitewines.iloc[:,:-1]
x
y = whitewines.iloc[:,-1]
y

x_train, x_test, y_train, y_test = train_test_split(x,y)

x_train.count()
x_test.count()

model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
model.fit(x_train, y_train)
result = model.predict(x_test)

sum(result == y_test) / len(y_test)
model.score(x_train, y_train)
model.score(x_test, result)
model.score(x_test, y_test)

from pandas import Series, DataFrame
max_depth1 = DataFrame(columns = ['train score','test score'])
max_depth1
for i in range(4,50):
    print("max_depth = {} ".format(i))
    model = DecisionTreeClassifier(criterion = 'entropy', max_depth = i)
    model.fit(x_train, y_train)
    result = model.predict(x_test)

    (result == y_test).value_counts()
    
    
    s = model.score(x_train, y_train)
    max_depth1.at[i,'train score'] = s
    p = model.score(x_test, y_test)
    max_depth1.at[i,'test score'] = p
    
max_depth2 = DataFrame(columns = ['train score','test score'])
max_depth2
for i in range(4,50):
    print("max_depth = {} ".format(i))
    model = DecisionTreeClassifier(criterion = 'gini', max_depth = i)
    model.fit(x_train, y_train)
    result = model.predict(x_test)

    (result == y_test).value_counts()
    s = model.score(x_train, y_train)
    max_depth2.at[i,'train score'] = s
    p = model.score(x_test, y_test)
    max_depth2.at[i,'test score'] = p
    
#위의 2가지 모델은 그리 결과가 좋게 나오지 않았다.
    
model1 = RandomForestClassifier(n_estimators = 10,
                               oob_score = True,
                               random_state = 0)

model1.fit(x_train, y_train)
result1 = model1.predict(x_test)
model1.score(x_train, y_train)
model1.score(x_test, y_test)
model1.feature_importances_
x.columns
importances = pd.Series(model1.feature_importances_,
                           index = x.columns)
importances

import matplotlib.pyplot as plt
import seaborn as sns

plt.Figure(figsize=(10,10))
plt.title("feature importance")
sns.barplot(x = importances, y = importances.index)


max_depth3 = DataFrame(columns = ['train score','test score'])
max_depth3

for i in range(5,50):
    model1 = RandomForestClassifier(n_estimators = i,
                               oob_score = True,
                               random_state = 0)

    model1.fit(x_train, y_train)
    result1 = model1.predict(x_test)
    s = model1.score(x_train, y_train)
    max_depth3.at[i,'train score'] = s
    p = model1.score(x_test, y_test)
    max_depth3.at[i,'test score'] = p
#
whitewines['quality'].value_counts()
#6    2198
#5    1457
#7     880
#8     175
#4     163
#3      20
#9       5
#Name: quality, dtype: int64
#점수대가 고르지 않고 편향되어 있다. 이를 2 ~ 3가지로 바꾸면 어떨까?
#때때로 종속변수의 분포를 조정해야 할 경우가 있다. 

#whitewines['quality'] = [0 if i < 4 2 elif i > 7 else 1 
#          for i in whitewines.iloc[-1]]

for i in range(len(whitewines)):
    if whitewines.iloc[i,-1] >= 7:
        whitewines.iloc[i,-1] = 2
    elif whitewines.iloc[i,-1] <= 4:
        whitewines.iloc[i,-1] = 0
    else:
        whitewines.iloc[i,-1] = 1
        
whitewines['quality'].value_counts()   

max_depth4 = DataFrame(columns = ['train score','test score'])
max_depth4

x = whitewines.iloc[:,:-1]
x
y = whitewines.iloc[:,-1]
y

x_train, x_test, y_train, y_test = train_test_split(x,y)

for i in range(5,50):
    model1 = RandomForestClassifier(n_estimators = i,
                               oob_score = True,
                               random_state = 0)

    model1.fit(x_train, y_train)
    result1 = model1.predict(x_test)
    s = model1.score(x_train, y_train)
    max_depth4.at[i,'train score'] = s
    p = model1.score(x_test, y_test)
    max_depth4.at[i,'test score'] = p
    
#정답률이 올라감 
    
plt.plot(max_depth4)

###### 선생님의 풀이 #######
#회사에서 원하는 것, 다양한 방법으로 시도해서 정답률을 높이기 
#데이터를 직접 조정을 할 수도 있다. 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

wine = pd.read_csv("C:\WorkSpace\Python_Space\data\whitewines.csv")


x = wine.drop("quality", axis=1)
y = wine["quality"]
y.value_counts()

# DecisionTree 학습하기 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = DecisionTreeClassifier(criterion='entropy',max_depth=10)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))


# RandomForest 학습하기 
model = RandomForestClassifier(n_estimators=100,oob_score=True)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))


# RandomForest 학습하기
model = RandomForestClassifier(criterion='entropy',n_estimators=500,oob_score=True)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))

############################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


wine = pd.read_csv("c:/data/whitewines.csv")

y = wine["quality"]
x = wine.drop("quality", axis=1)

lst = []
for v in list(y):
    if v <= 4:
        lst += [0]
    elif v <= 7:
        lst += [1]
    else:
        lst += [2]
y = lst



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import collections
collections.Counter(y_train)

model = RandomForestClassifier()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))



wine = pd.read_csv("c:/data/whitewines.csv")

y = wine["quality"]
x = wine.drop("quality", axis=1)

y = [0 if i <=6 else 1 for i in y]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import collections
collections.Counter(y_train)
collections.Counter(y_test)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))


# LogisticRegression
 
import pandas as pd
from sklearn.linear_model import LogisticRegression

wine = pd.read_csv("c:/data/whitewines.csv")
y = wine["quality"]
x = wine.drop("quality", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import collections
collections.Counter(y_train)
collections.Counter(y_test)
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))





wine = pd.read_csv("c:/data/whitewines.csv")

y = wine["quality"]
x = wine.drop("quality", axis=1)

y = [0 if i <=6 else 1 for i in y]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import collections
collections.Counter(y_train)
collections.Counter(y_test)

logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))




wine = pd.read_csv("c:/data/whitewines.csv")

y = wine["quality"]
x = wine.drop("quality", axis=1)

lst = []
for v in list(y):
    if v <= 4:
        lst += [0]
    elif v <= 7:
        lst += [1]
    else:
        lst += [2]
y = lst

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))

#스케일 작업이 필요한가???
#scale 해보자 
whitewine.



############
#credit file을 읽어들여보자 
credit = pd.read_csv("C:\WorkSpace\Python_Space\data\credit.csv")

x = credit.drop('default',axis = 1)
y = credit['default']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier(n_estimators = 100, oob_score = True)
model.fit(x_train, y_train)
#독립변수와 종속변수가 모두 수치형이 아니라서 작동하지 않는다. 

x.info()
x = pd.get_dummies(x)
x.info()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier(n_estimators = 100, oob_score = True)
model.fit(x_train, y_train)
result = model.predict(x_test)
model.score(x_train, y_train)
model.score(x_test, y_test)
