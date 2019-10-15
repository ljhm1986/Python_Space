# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:38:45 2019

@author: stu11
"""

#[문제 178] KNN 프로그램을 만들어 보자

pointlist = [(1,1),(1,0),(2,0),(0,1),(2,2),(1,5),(2,3)]

def knn(point, dataset, number):
    
    distanceSet = {}
    result = []
    
    #두 점의 거리를 계산하고, 점과 거리를 한 쌍으로 저장한다.
    for i in dataset:
        distance_temp = ((point[0] - i[0])**2 + (point[1] - i[1])**2)**(1/2)
        distanceSet[i] = distance_temp
      
    #dictionary를 거리크기순으로 정렬한다. 
    import operator     
    data_sort = sorted(distanceSet.items(), key=operator.itemgetter(1))
    print(type(data_sort))#list
    
    #거리가 가까운 점들을 출력한다.
    for i in range(0,number):
        result.append(data_sort[i][0])
        
    return result    
        
knn((2,3),pointlist,2)

## 선생님의 풀이 ##
import numpy as np
import operator 
point = [2,3]

#브로드케스팅 되어서 array의 크기가 달라도 된다. 이것을 거리 계산에 사용하자 
np.array(point) - np.array(pointlist)
np.sqrt(np.sum(pow(np.array(point) - np.array(pointlist),2),axis = 1))

def knn(point, pointlist, k):
    
    dic = {}
    
    for p in pointlist:
        d = np.sqrt(np.sum(pow(np.array(point) - 
                               np.array(p),2)))
        dic[p] = d
    
    sort_dic = sorted(dic.items(), key=operator.itemgetter(1))
    print(sort_dic)
    res = []
    
    #key, value중 key 만 필요하니까 
    for key in sort_dic:
        if len(res) < k:
            res.append(key[0])
    
    return res

###
dist = np.sqrt(np.sum(pow(np.array(point) - np.array(pointlist),2),axis = 1))
dist
#오름차순된 index번호를 반환한다. 
indices = dist.argsort()
#2개만 보려면 
indices = dist.argsort()[:2]
#내림차순으로 보려면 
indices = dist.argsort()[::-1]

print(indices)
type(dist.argsort())#array

#그럼 pointlist에서 정리된 index 순으로 출력하자 
for i in indices:
    print(pointlist[i])

#다음과 같이 하면 안 된다. dist.argsort()가 array라서 
pointlist[dist.argsort()]

#take()를 이용해보자 
np.take(pointlist,indices,axis = 0)
np.take(pointlist,indices,axis = 0)[0:2]

#####
#[문제 179] food.csv 읽어 들인 후 토마토 단맛 6, 아삭한 맛 4를
#이용해서 거리계산한 값을 dist 컬럼을 추가하고 dist 컬럼의 값을
#기준으로 오름차순 순위 컬럼 rank를 생성하세요.

import csv
file = open('C:\\WorkSpace\\Python_Space\\data\\food.csv','r')
food_csv = csv.reader(file)

#list에 데이터 저장 
list1 = []
for i in food_csv:
    list1.append(i)

file.close()

list1
list1[1:]
list1[1][1]

point = (6,4)

list1 = list1[1:]
for i in range(len(list1)):
    print(list1[i][1:3])
    distance_temp = ((point[0] - int(list1[i][1]))**2 
                     + (point[1] - int(list1[i][2]))**2)**(1/2)
    list1[i].append(distance_temp)

list1
#정렬은...

dict1 = {}
for i in list1:
    dict1[i[0]] = i[4]
    
dict1

sort_dict1 = sorted(dict1.items(), key=operator.itemgetter(1))
sort_dict1

for i in sort_dict1[:2]:
    print(i[0])

#pandas 를 이용해서 하자 
import pandas as pd
food_csv2 = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\food.csv')

food_csv2

#x_train = np.array(df.iloc[:,1:3])
array1 = np.array([[6,4]]) - np.array(food_csv2[['sweetness','crunchiness']])
#array1 = np.array([[6,4]]) - x_train

dist = np.sqrt(np.sum(pow(array1,2),axis = 1))
dist

food_csv2['dist'] = dist
food_csv2
food_csv2['dist'].rank(method = 'dense')
food_csv2['rank'] = food_csv2['dist'].rank(method = 'dense')
food_csv2.info()

#k 값은 짝수 선호 
#k = 1, 과일로 분류
#k = 3, 과일이 더 많으니까 과일로 분류 
#같은 갯수가 나오면 거리 더한게 더 짧은것으로 
food_csv2['rank'] <= 3
food_csv2[food_csv2['rank'] <= 3]['class']
food_csv2[food_csv2['rank'] <= 3]['class'].value_counts()

import collections
#most_common() : 갯수를 해아린 다음에 큰 순서대로만 뽑음 
count = collections.Counter(food_csv2[food_csv2['rank'] <= 3]['class'])
count.most_common(1)#[('Fruits', 2)]
count.most_common(1)[0][0]#'Fruits'

c = collections.Counter('apple, orange')
c
c.most_common()
c.most_common(3)

#지금까지 한 과정을 라이브러리를 불러와서 해 보자 
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\food.csv')

#훈련 데이터 셋
x_train = np.array(df.iloc[:,1:3])
#분류기준 
label = df['class']

#근접 점의 갯수
clf = KNeighborsClassifier(n_neighbors = 3)
#훈련시키기 
clf.fit(x_train,label)
#이제 새로운 점을 넣어보자 
clf.predict(np.array([[6,4]]))[0]

#[문제180] 키, 몸무게에 따른 성별을 분류해주세요.
#
#키, 몸무게 데이터
#[[158, 64],
#[170, 86],
#[183, 84],
#[191, 80],
#[155, 49],
#[163, 59],
#[180, 67],
#[158, 54],
#[170, 67]]
#
#성별 레벨
#['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female']
#
#
#[155, 70] 성별을 분류하세요.
#'female'
#라이브러리를 사용하면 
from sklearn.neighbors import KNeighborsClassifier
x_train = [[158,64],[170,86],[183,84],[191,80],
           [155,49],[163,59],[180,67],[158,54],[170,67]]

label = ['male', 'male', 'male', 'male',
         'female', 'female', 'female', 'female', 'female']

clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(x_train, label)
clf.predict(np.array([[155,70]]))[0]

#라이브러리를 사용하지 않고 class를 만들어서 해봄 
class knn_knn:
    import pandas as pd
    
    def __init__ (self,k_n):
        self.n_neighbers = k_n
        
    def fit(self, x_train, label):
        from pandas import Series, DataFrame
        self.df = DataFrame(x_train)
        self.df[2] = label
            
    def predict(self,x):
        import numpy as np
        array1 = np.array(x) - np.array(df[[0,1]])
        dist = np.sqrt(np.sum(pow(array1,2),axis = 1))
        df['dist'] = dist
        df['rank'] = df['dist'].rank(method = 'dense')
        
        import collections
        count = collections.Counter(df[df['rank'] <= self.n_neighbers][2])
        
        return count.most_common(1)[0][0]
    
knn_knn = knn_knn(3)        
knn_knn.fit(x_train,label)
knn_knn.predict([155,70])

### 선생님의 풀이 중에서 ###
#다음과 같이 넣어두면 1행 2열이 된다. 
y = np.array([155,70])
y.shape
#다음과 같이 해야 1행 1열이 된다. 
y = np.array([[155,70]])
y.shape

##아까 했던 food를 다시 보면 
food2  =pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\food.csv')
food2['sweetness']
food2['crunchiness']

dist = np.sqrt((food2['sweetness'] - 6)**2 + (food2['crunchiness']-4)**2)
dist
#두 dataframe을 합치자 
data = pd.merge(food2, pd.DataFrame({'dist':dist}),
                left_index = True, right_index = True)

#dist column을 기준으로 정렬하자 
data.sort_values(by='dist',ascending = True)
data.sort_values(by='dist',ascending = False)

#rank column을 추가하자 
data['rank'] = data['dist'].rank(ascending = True, method = 'dense')
data

#이제 counter를 사용하지 않고 구해보자 
data.loc[data['rank'] <= 3]['class'].value_counts()
#값이 큰것만 고르기 
data.loc[data['rank'] <= 3]['class'].value_counts().max()

#x는 Series, dictionary 형태이다. 
x = data.loc[data['rank'] <= 3]['class'].value_counts()[
        data.loc[data['rank'] <= 3]['class'].value_counts() == 
        data.loc[data['rank'] <= 3]['class'].value_counts().max()
        ]
x.keys()[0]

