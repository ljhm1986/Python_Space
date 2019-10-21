# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:38:45 2019

@author: stu11
"""
###################################################################
#10/15#
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

####################################################################
#10/16#
#[문제 182] 나이, 월수입, 상품구매 여부 3개의 데이터를 이용해서
#나이가 44이고 월급이 400만원인 사람이 상품을 구매할지 
#비구매할지를 knn으로 분류해주세요 
import pandas as pd
buy_data = pd.read_csv("C:\\WorkSpace\\R_Space\\data\\buy.csv")
buy_data

x_train = buy_data[['나이','월수입']]
label = buy_data['상품구매여부']

from sklearn.neighbors import KNeighborsClassifier
buy_predict = KNeighborsClassifier(3)
buy_predict.fit(x_train,label)
buy_predict.predict([[44,400]])
buy_predict.predict([[44,400]])[0]
#'구매'

#####
import pandas as pd
import numpy as np

x = (np.arange(9, dtype = np.float) - 3)
x
x.shape
x = x.reshape(-9,1)
x
np.vstack([x, [100]])#100이 추가된다.
x
pd.DataFrame(x).describe()

#표준정규분호로 표준화하는 방법 
y = (x - np.mean(x))/np.std(x)

pd.DataFrame(y).describe()

from sklearn.preprocessing import StandardScaler
#평균 0, 분산 1로 조정된다.
y_1  = StandardScaler().fit_transform(x)

np.mean(y_1)
np.std(y_1)

scaler = StandardScaler()
scaler.fit(x)
y_2 = scaler.transform(x)

#최소값과 최대값을 사용해서 0 ~ 1 사이의 데이터로 표준화 방법
(x - x.min())/(x.max() - x.min())

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
min_max_scaler.fit_transform(x)

from sklearn.preprocessing import minmax_scale
minmax_scale(x)

#평균 0, 분산 1로 조정하는 방법 
from sklearn.preprocessing import scale
np.mean(scale(x))
np.std(scale(x))

#[문제 185] bmi데이터를 이용해서 키: 178, 몸무게 71 일때 분류해조세요

bmi_data = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\bmi.csv")
bmi_data
bmi_data.info()

#scale 하지 않고 knn을 수행하자 
height_data = bmi_data['height']
weight_data = bmi_data['weight']
label_data = bmi_data['label']
x_train = bmi_data[['height','weight']]

test_data = np.array([[178,71]])
test_data2 = np.array([[155,69]])

bmi_test = KNeighborsClassifier(5)
bmi_test.fit(x_train, label_data)
bmi_test.predict(test_data)
#normal
bmi_test.predict_proba(test_data)
bmi_test.predict(test_data2)
#fat

#정규화를 한 뒤에 해 보자 
from pandas import DataFrame
bmi_data_scale = DataFrame()

#x_train 을 정규화 해 보자 
height_mean = bmi_data['height'].mean()
height_std = bmi_data['height'].std()
bmi_data_scale['height'] = (bmi_data['height'] - height_mean)/height_std

weight_mean = bmi_data['weight'].mean()
weight_std = bmi_data['weight'].std()
bmi_data_scale['weight'] = (bmi_data['weight'] - weight_mean)/weight_std

x_train_scale = bmi_data_scale[['height','weight']]

#test 를 정규화 해 보자 
test_data_scale = np.array([[0.0,0.0]])
test_data_scale[0][0] = (test_data[0][0] - height_mean)/height_std
test_data_scale[0][1] = (test_data[0][1] - weight_mean)/weight_std

test_data2_scale = np.array([[0.0,0.0]])
test_data2_scale[0][0] = (test_data2[0][0] - height_mean)/height_std
test_data2_scale[0][1] = (test_data2[0][1] - weight_mean)/weight_std

#정규화 한 데이터를 넣어서 수행하자 
bmi_test_scale = KNeighborsClassifier(5)
bmi_test_scale.fit(x_train_scale, label_data)
bmi_test_scale.predict(test_data_scale)
#normal
bmi_test_scale.predict(test_data2_scale)
#fat
bmi_test_scale.predict_proba(test_data_scale)

#minmax_scaler 를 한 뒤에 해보자 
bmi_data_maxmin = DataFrame()

#x_train을 minmax 하자 
height_max = bmi_data['height'].max()
height_min = bmi_data['height'].min()
bmi_data_maxmin['height'] = (bmi_data['height'] - height_min)/(
                                height_max - height_min)

weight_max = bmi_data['weight'].max()
weight_min = bmi_data['weight'].min()
bmi_data_maxmin['weight'] = (bmi_data['weight'] - weight_min)/(
                                weight_max - weight_min)

x_train_maxmin = bmi_data_maxmin[['height','weight']]

#text 를 minmax 하자 
test_data_maxmin = np.array([[0.0,0.0]])
test_data_maxmin[0][0] = (test_data[0][0] - height_max)/(
                            height_max - height_min)
test_data_maxmin[0][1] = (test_data[0][1] - weight_max)/(
                            weight_max - weight_min)

test_data2_maxmin = np.array([[0.0,0.0]])
test_data2_maxmin[0][0] = (test_data2[0][0] - height_min)/(
                            height_max - height_min)
test_data2_maxmin[0][1] = (test_data2[0][1] - weight_min)/(
                            weight_max - weight_min)

#minmax 한 것을 knn으로 수행하자 

bmi_test_maxmin = KNeighborsClassifier(5)
bmi_test_maxmin.fit(x_train_maxmin, label_data)
bmi_test_maxmin.predict(test_data_maxmin)
#normal
bmi_test_maxmin.predict(test_data2_maxmin)
#fat
bmi_test_maxmin.predict_proba(test_data_maxmin)

#이번 경우에는 3가지 방법 모두 같은 값이 나왔다.
#3가지 방법으로 했는데 값이 다르게 나오는 경우가 있다. 그런 경우에는 직접
#데이터 셋에서 샘플을 뽑아서 확인하거나, 데이터의 label이 잘 되어 있는지 
#확인해야 한다. 

#### 선생님의 풀이 ####
#[문제185] bmi 데이터를 이용해서 키 : 178, 몸무게 : 71 일때 분류해주세요.
bmi = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\bmi.csv")
bmi.info()

#scale 안 하고 함 
x_train = np.array(bmi.iloc[:,:2])
x_train
label = bmi['label']
label
y = np.array([[155, 69]])

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,label)
clf.predict(y)[0]#fat

##standard scale
bmi = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\bmi.csv")
bmi.info()
x_train = np.array(bmi.iloc[:,:2])
x_train

h_mean =  np.mean(bmi.iloc[:,0])
h_std  = np.std(bmi.iloc[:,0])

w_mean =  np.mean(bmi.iloc[:,1])
w_std  = np.std(bmi.iloc[:,1])

from sklearn.preprocessing import scale
x_train = np.array(scale(bmi.iloc[:,:2]))
x_train
label = bmi['label']
label
y = np.array([[155,69]])
h_s = (y[0][0]-h_mean) / h_std
w_s = (y[0][1]-w_mean) / w_std
test = np.array([[h_s,w_s]])
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,label)
clf.predict(test)[0]#fat

#minmax scale
bmi = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\bmi.csv")

bmi.info()
x_train = np.array(bmi.iloc[:,:2])

h_min =  np.min(bmi.iloc[:,0])
h_max  = np.max(bmi.iloc[:,0])

w_min =  np.min(bmi.iloc[:,1])
w_max  = np.max(bmi.iloc[:,1])

from sklearn.preprocessing import minmax_scale

x_train = np.array(minmax_scale(bmi.iloc[:,:2]))
x_train
label = bmi['label']
label
y = np.array([[155,69]])
h_minmax = (y[0][0]-h_min) / (h_max - h_min)
w_minmax = (y[0][1]-w_min) / (w_max - w_min)
test = np.array([[h_minmax,w_minmax]])
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,label)
clf.predict(test)[0]#fat

##height는 200으로 나눔, weight는 100으로 나눔 
bmi = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\bmi.csv")
bmi
bmi.info()

x_train = np.array([bmi['height']/200,bmi['weight']/100]).T
x_train

label = bmi['label']
label
y = np.array([[155/200,69/100]])

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,label)
clf.predict(y)[0]#fat

################################################################
#[문제186] 유방암 데이터 입니다. kNN알고리즘을 이용해서 훈련데이터셋,
#테스트데이터셋을 이용해서 분류가 잘되는지 확인하세요.
#
#
#1 단계 : 데이터 수집
#https://archive.ics.uci.edu/ml/machine-learning-databases/
#breast-cancer-wisconsin/wdbc.names
#
#- 위스콘신대학의 연구원들의 자료
#- 유방 종양의 미세침 흡인물 디지털 이미지에서 측정한 값 이며 이 값은 디지털 
#이미지에 나타난 세포 핵의 특징이다.
#- 암조직 검사에 대한 관측값은 569개, 변수(속성) 32
#- 식별숫자, 암진단 여부(악성(Malignant),양성(Benign)), 30개 수치 측정치
#- 세포핵의 모양과 크기관련된 10개 특성
#* radius(반지름)
#* texture(질감)
#* perimeter(둘레)
#* area(넓이)
#* smoothness(매끄러움)
#* compactness(조밀성)
#* concavity(오목함)
#* concave points(오목점)
#* symmetry(대칭성)
#* fractal dimension(프랙탈 차원)

#이런 데이터를 분석하려면 유방암에 대한 공부를 해야 한다.

breast_cancer = pd.read_csv(
        "C:\\WorkSpace\\Python_Space\\data\\wisc_bc_data.csv")

breast_cancer
breast_cancer.info()
#보면 column이 32개나 된다, 분석을 할때 id는 빼도 무방하다.  
#diagnosis는 라벨이다. 

#id column를 빼자 
breast_cancer = breast_cancer.iloc[:,1:]

breast_cancer['diagnosis'] 
#빈도수를 계산하자 
from collections import Counter
Counter(breast_cancer['diagnosis'])
#Counter({'B': 357, 'M': 212})
cn = Counter(breast_cancer['diagnosis'])
(cn['B'] / 569) * 100#62.741652021089635
(cn['M'] / 569) * 100#37.258347978910365

breast_cancer.describe()

#각 column마다 수치단위가 다르니까 scale 작업을 하자 
from sklearn.preprocessing import scale
breast_cancer.iloc[:,1:] = np.array(scale(breast_cancer.iloc[:,1:]))
breast_cancer.describe()
breast_cancer.head(5)

#70%를 학습하고, 30%는 시험해보자 
#train_data에다가 label이 비율을 맞추어서 추출하자 
from sklearn.model_selection import train_test_split
X = breast_cancer.iloc[:,1:]
Y = breast_cancer.iloc[:,0]#label

#80%는 훈련 데이터, 20%는 시험 데이터
#train_test_split() 시행할때 마다 추출하는 샘플이 다르다 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
#대략 비율 좋게 나왔다.
(Counter(Y_train)['B'] * 100) / (Counter(Y_train)['B'] + Counter(Y_train)['M'])
(Counter(Y_test)['B'] * 100) / (Counter(Y_test)['B'] + Counter(Y_test)['M'])

#knn에 이웃수는 몇으로 할까? 
np.sqrt(569)
clf = KNeighborsClassifier(n_neighbors = 23)
clf.fit(X_train, Y_train)
Counter(clf.predict(X_test) == Y_test)#Counter({True: 108, False: 6})
108/ 114
#score() 정답인 비율을 보여준다. 
clf.score(X_test, Y_test)
clf.score(X_test, Y_test) == 108/ 114#True
#그럼 틀리게 나온 데이터를 찾아보자, 그리고 진단이 잘 되었는지 의뢰해 보자 

X_test[clf.predict(X_test) != Y_test]

idx = X_test[clf.predict(X_test) != Y_test].index.tolist()
idx
breast_cancer.iloc[idx]
breast_cancer.iloc[idx]['diagnosis']
#예측이 잘못 되었으면 k값을 조정해야 한다.
#또는 다른 scale 조정 방법을 택한다.

###############################################################
#10/17#
#어제 한 유방암 데이터를 다시 살펴 봄 
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#score () : 예측된 값과 실제 데이터의 값의 일치비율 
clf.score(X_test, Y_test)

#혼동행렬(confusion matrix) : 모델 성능을 평가할 때 사용되는 지표
#예측값이 실제 관측값을 얼마나 정확히 예측했는지 보여주는 행렬
from sklearn.metrics import classification_report, confusion_matrix
y_predict = clf.predict(X_test)
Y_test.shape
y_predict.shape

#예측한 결과와 실제 값을 비교해 보자 
#test 값과 예측결과를 넣어야 한다.
print(confusion_matrix(Y_test,y_predict))
#    B  M
#B [67 0 ]
#M [ 8 39]
from collections import Counter
Counter(y_predict)#Counter({'M': 39, 'B': 75}) 예측된 갯수
Counter(Y_test)#Counter({'M': 47, 'B': 67}) 실제 갯수 
#예로 다음과 같이 2가지 결과가 나왔다고 하자 
############################
#[A]             예측
#실제       암환자    일반환자
#---------------------------
#암환자         9          1
#일반환자      30         60
############################
#[B]             예측
#실제       암환자    일반환자
#---------------------------
#암환자         1          9
#일반환자      20         70
###########################
#위 둘을 비교하면 A가 B보다 더 좋게 예측을 한다 

#            예상(예)   예상(아니오)
#----------------------------
#실제(예)         TP       FN
#실제(아니오)      FP       TN
#
#TP(True Positive) :참 긍정
#병에 관한 아니오 라고 예측한 환자가 실제로 병이 없는 경우
#TN(True Negative) : 참 부정
#병에 관해 예 라고 예측한 환자가 실제로 병이 있는 경우
#FP(False Positive) : 거짓 긍정
#병에 관해 예 라고 예측한 환자가 실제로 병이 없는 경우 
#FN(False Negative) : 거짓 부정 
#병에 관해 아니오 라고 예측한 환자가 실제로 병이 있는 경우

#예측 모델의 정확도(accuracy)
#모델이 입력된 데이터에 대해 얼마나 정확하게 예측하는지를 나타낸다.
#정확도 = 예측결과와 실제값이 동일한 건수 / 전체 데이터 수 
#      = (TP + FN) / (TP + TN + FP + FN)
(60 + 9) / 100#0.69
(70 + 1) / 100#0.71
#B의 정확도가 더 높다.

#정밀도(precision) : positive 로 예측된 결과의 정확도
#TP / (TP + FP)
9 / (9 + 30)#0.230
1 / (1 + 20)#0.0476
#A의 정밀도가 더 높다. 
print(classification_report(Y_test, y_predict)) 
#              precision    recall  f1-score   support
#
#           B       0.89      1.00      0.94        67
#           M       1.00      0.83      0.91        47
#
#    accuracy                           0.93       114
#   macro avg       0.95      0.91      0.93       114
#weighted avg       0.94      0.93      0.93       114

#재현율(recall) : 실제 positive중 positive로 예측한 비율 
#TP / (TP + FN)
#실제값 중에서 모델이 검출한 실제값의 비율을 나타내는 지표
#실제로 병이 있는 전체중 참 긍정 비율?
#실제 암환자들이 병원에 갔을때 암환자라고 예측될 확률, 
#조기에 정확하게 발견해서 신속하게 처방하는 것이 올바른 모델
9 / (1 + 9)#0.9
1 / (1 + 9)#0.1

#f1-score 
#정밀도도 중요하고 재현율도 중요한데 둘 중 무엇을 쓸지 고민될 수 있다.
#이 두값의 조화평균을 내서 하나의 수치로 나타낸 지표 
#f1-score : 2 * 평균 재현율 * 평균 정밀도 / (평균 재현율 + 평균 정밀도)

#B 정밀도 : 67 / (67 + 8)
#M 정밀도 : 40 / (40 + 0)
#B 재현율 : 67 / (67 + 0)
#M 재현율 : 40 / (40 + 6)

### iris data를 불려들여봐서 knn를 해 보자 ###
iris_data = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\iris.csv")
iris_data
iris_data.info()

#빈도수를 계산하자
from collections import Counter
Counter(iris_data['Name'])

#scale  하지 않고 해봄
from sklearn.model_selection import train_test_split
X = iris_data.iloc[:,0:4]
Y = iris_data.iloc[:,4]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(21)
clf.fit(X_train, Y_train)
clf.predict(X_test)
#예측이 맞는 비율 
clf.score(X_test,Y_test)
#예측과 실제값이 다른 row
X_test[clf.predict(X_test) != Y_test]

from sklearn.metrics import classification_report, confusion_matrix
Y_predict = clf.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(classification_report(Y_test, Y_predict)) 

#standard scale을 하고 해 보자 
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(iris_data.iloc[:,0:4])
#from sklearn.preprocessing import scale
#X = scale(iris_data.iloc[:,0:4])
X = scale(iris_data.iloc[:,0:4])
Y = iris_data.iloc[:,4]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(21)
clf.fit(X_train, Y_train)
clf.predict(X_test)
#예측이 맞는 비율 
clf.score(X_test,Y_test)
#예측과 실제값이 다른 row
X_test[clf.predict(X_test) != Y_test]

#minmax scale을 하고 해 보자 
from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(iris_data.iloc[:,0:4])
from sklearn.preprocessing import minmax_scale
X = minmax_scale(iris_data.iloc[:,0:4])

iris_data.columns
bool(iris_data.iloc[:,5])
len(iris_data)

df = DataFrame(index = range(len(iris_data)),
               columns = iris_data.columns)

def standard_scaler(dataFrame):
    
    from pandas import Series, DataFrame
    
    df = DataFrame(index = range(len(dataFrame)),
               columns = dataFrame.columns)
    
    i = 0
    while(True):
        
        try:
            mean = dataFrame.iloc[:,i].mean()
            std = dataFrame.iloc[:,i].std()
            for j in range(len(dataFrame)):
                df.iloc[j,i] = (dataFrame.iloc[j,i] - mean) / std
        except:
            break
        i += 1
    
    return df

X = standard_scaler(iris_data.iloc[:,0:4])

def minmax_scaler(dataFrame):
    
    from pandas import Series, DataFrame
    
    df = DataFrame(index = range(len(dataFrame)),
               columns = dataFrame.columns)
    
    i = 0
    while(True):
        
        try:
            max_value = dataFrame.iloc[:,i].max()
            min_value = dataFrame.iloc[:,i].min()
            for j in range(len(dataFrame)):
                df.iloc[j,i] = (dataFrame.iloc[j,i] - min_value) / \
                                (max_value - min_value)
        except:
            break
        i += 1
    
    return df

X = minmax_scaler(iris_data.iloc[:,0:4])
Y = iris_data.iloc[:,4]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(21)
clf.fit(X_train, Y_train)
clf.predict(X_test)
#예측이 맞는 비율 
clf.score(X_test,Y_test)
#예측과 실제값이 다른 row
X_test[clf.predict(X_test) != Y_test]

#그럼 k 가 1~100일때 예측이 얼마나 맞는지 보자 
score_data = []
for i in range(1,101):
    clf = KNeighborsClassifier(i)
    clf.fit(X_train, Y_train)
    score = clf.score(X_test,Y_test)
    print("이웃이 {}일때 정확도는 {} 이다.".format(i,score))
    score_data.append(score)
    
score_data
    
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

plt.plot(score_data)
plt.title("k값에 따른 예측이 맞는 비율")
plt.xlabel("k")
plt.ylabel("예측율")
