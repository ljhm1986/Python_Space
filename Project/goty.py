import pandas as pd
import numpy as np 
from pandas import Series, DataFrame

#3등 이내 2점, 2개 이상 1점, 그외 0점 

#2013
goty2013A = ['The Last of Us', 'Grand Theft Auto V', 'BioShock Infinite']
goty2013B= ['Super Mario 3D World', 'Gone Home', 'Tome Raider',
 'The Legend of Zelda: A Link Between Worlds',
 "Assassin's Creed IV: Black Flag", 'Papers, Please', 
 'Rayman Legends', 'Brothers: A Tale of Two Sons', 'The Stanley Parable',
 'Pokémon X·Y', 'Battlefield 4', 'Ni no Kuni: Wrath of the White Witch']
#2014
goty2014A= ['Dragon Age: Inquisition', 'Middle-earth: Shadow of Mordor',
'Mario Kart 8', 'Super Smash Bros.']#공동 3등
goty2014B= ['Far Cry 4','Alien: Isolation','Bayonetta 2', 'Dark Souls II', 'Destiny',
 'Hearthstone: Heroes of Warcraft', 'Kentucky Route Zero - Act III',
 'Sunset Overdrive', 'Divinity: Original Sin', 'Forza Horizon 2', 
 'Wolfenstein: The New Order', 'This War of Mine', 'Titanfall',
 'Call of Duty: Advanced Warfare','Danganronpa: Trigger Happy Havoc',
 'Shovel Knight', 'South Park: The Stick of Truth', 'The Evil Within', 
 'The Walking Dead: Season Two', 'Transistor']
#2015
goty2015A= ['The Witcher 3: Wild Hunt', 'Fallout 4', 'Bloodborne']
goty2015B= ['Metal Gear Solid V: The Phantom Pain', 'Life Is Strange', 'Super Mario Maker',
 'Undertale', 'Rocket League', 'Rise of the Tomb Raider', 'Batman: Arkham Knight',
 'Her Story', "Assassin's Creed Syndicate", 'Splatoon']
#2016
goty2016A= ["Uncharted 4: A Thief's End", 'Overwatch', 'DOOM']
goty2016B= ['Battlefield 1', 'The Last Guardian', 'INSIDE', 'Dark Souls III','Final Fantasy XV',
 'The Witcher 3: Wild Hunt - Blood and Wine', 'Dishonored 2', 'The Witness',
 'Titanfall 2', 'Hitman', 'Pokémon GO', 'XCOM 2', 'SUPERHOT']
#2017
goty2017A= ['The Legend of Zelda: Breath of the Wild', 'Horizon Zero Dawn',
            'Super Mario Odyssey']
goty2017B= ['NieR: Automata', 'Persona 5', "PlayerUnknown's Battlegrounds",
 "Assassin's Creed Origins", 'Divinity: Original Sin II', 'Wolfenstein II: The New Colossus',
 'Prey', 'Resident Evil 7: Biohazard', "Hellblade: Senua's Sacrifice"]
#2018
goty2018A= ['God of War', 'Red Dead Redemption 2', "Marvel's Spider-Man"]
goty2018B= ['Celeste', 'Return of the Obra Dinn', 'Fortnite Battle Royale',
            'Monster Hunter: World',
 'Super Smash Bros. Ultimate', 'Tetris Effect', 'Astro Bot Rescue Mission',
 'Florence', 'Kingdom Come: Deliverance']

gotyA = []
gotyA = goty2013A + goty2014A + goty2015A + goty2016A + goty2017A + goty2018A
gotyB = []
gotyB = goty2013B + goty2014B + goty2015B + goty2016B + goty2017B + goty2018B

len(gotyA)#19
len(gotyB)#73
#count = 0
#for i in metascoreDF['name']:
#    if i in gotyA:
#        count += 1
#print(count)#39
#
#count = 0
#for i in opencriticDF['name']:
#    if i in gotyA:
#        count += 1
#print(count)#15
#
#count = 0
#for i in metascoreDF['name']:
#    if i in gotyB:
#        count += 1
#print(count)#153
#
#count = 0
#for i in opencriticDF['name']:
#    if i in gotyB:
#        count += 1
#print(count)#53
#
#metascoreDF[metascoreDF['name'].isin(gotyA)]


mergeDF = pd.read_csv("C:/WorkSpace/Python_Space/Project/merge.csv")
mergeDF = mergeDF.iloc[:,1:]
mergeDF.info()

mergeDF = mergeDF[['name','year_x','metascore','userscore','openscore',
                   'action','fighting','first-person','flight','party',
                   'puzzle','racing','real-time','role-playing','simulation',
                   'sports','strategy','third-person','turn-based','wargame',
                   'wrestling']]

#k-mean 을 이용해서 여러 그룹으로 나누어 보자 
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 4)

#2가지 스케일링으로 실시해 본다.
## 표준화(standardization) ##
import copy
X = copy.deepcopy(mergeDF)
X.columns

from sklearn.preprocessing import StandardScaler
X_train = X.iloc[:,2:5]
X_train_standardScale = StandardScaler().fit_transform(X_train)

model.fit(X_train_standardScale)
model.labels_
model.cluster_centers_
collections.Counter(model.labels_)

colormatp = np.array(['red','blue','green','black'])
plt.scatter(mergeDF.iloc[:,2], mergeDF.iloc[:,3],
            c = colormatp[model.labels_], s = 2)
plt.show()

centers = pd.DataFrame(model.cluster_centers_)
plt.scatter(centers.iloc[:,0], centers.iloc[:,1], s = 50,
            marker = 'D', c='g')
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
axes3d = plt.axes(projection = '3d')
axes3d.scatter(X.iloc[:,2], X.iloc[:,3], X.iloc[:,4],
               c = colormatp[model.labels_], s = 2)
plt.show()

#응집도 
model.inertia_

l = range(1,11)
inertia = []
for k in l:
    model = KMeans(n_clusters = k)
    model.fit(X_train_standardScale)
    inertia.append(model.inertia_)
    
inertia

plt.plot(l, inertia, '-o')
plt.xlabel("number of cluster K")
plt.ylabel("inertia")
plt.xticks(l)
plt.show()

## 정규화(normalization) ##
model2 = KMeans(n_clusters = 4)

import copy
X = copy.deepcopy(mergeDF)
X.columns


from sklearn.preprocessing import MinMaxScaler
X_train = X.iloc[:,2:5]
type(scale(X_train))
X_train_normalScale = MinMaxScaler().fit_transform(X_train)

model2.fit(X_train_normalScale)
model2.labels_
model2.cluster_centers_
collections.Counter(model2.labels_)

colormatp = np.array(['red','blue','green','black'])
plt.scatter(mergeDF.iloc[:,2], mergeDF.iloc[:,3],
            c = colormatp[model2.labels_], s = 2)

centers = pd.DataFrame(model2.cluster_centers_)
plt.scatter(centers.iloc[:,0], centers.iloc[:,1], s = 50,
            marker = 'D', c='g')
plt.show()

fig = plt.figure()
axes3d = plt.axes(projection = '3d')
axes3d.scatter(X.iloc[:,2], X.iloc[:,3], X.iloc[:,4],
               c = colormatp[model2.labels_], s = 2)
plt.show()

#응집도 
model2.inertia_

l = range(1,11)
inertia = []
for k in l:
    model2 = KMeans(n_clusters = k)
    model2.fit(X_train_normalScale)
    inertia.append(model2.inertia_)
    
inertia

plt.plot(l, inertia, '-o')
plt.xlabel("number of cluster K")
plt.ylabel("inertia")
plt.xticks(l)
plt.show()

#두 스케일로 그룹이 달라지는가? 근데 각 위치별로 숫자가 일치하는건 아닌데
#... 

(model.labels_ == model2.labels_).sum()

## 이번에는 genre를 포함해서 k-mean을 해 보자 ###
model3 = KMeans(n_clusters = 4)

#2가지 스케일링으로 실시해 본다.
## 표준화(standardization) ##
import copy
X = copy.deepcopy(mergeDF)
X.columns

from sklearn.preprocessing import StandardScaler
X_train = X.iloc[:,2:]
X_train_standardScale = StandardScaler().fit_transform(X_train)

model3.fit(X_train_standardScale)
model3.labels_
model3.cluster_centers_
collections.Counter(model3.labels_)

X['cluster'] = model3.labels_
X['metascore'].groupby(X['cluster']).mean()
X['action'].groupby(X['cluster']).mean()

colormatp = np.array(['red','blue','green','black'])
plt.scatter(mergeDF.iloc[:,2], mergeDF.iloc[:,3],
            c = colormatp[model3.labels_], s = 2)
plt.show()

centers = pd.DataFrame(model3.cluster_centers_)
plt.scatter(centers.iloc[:,0], centers.iloc[:,1], s = 50,
            marker = 'D', c='g')
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
axes3d = plt.axes(projection = '3d')
axes3d.scatter(X.iloc[:,2], X.iloc[:,3], X.iloc[:,4],
               c = colormatp[model3.labels_], s = 2)
plt.show()

#응집도 
model3.inertia_

l = range(1,11)
inertia = []
for k in l:
    model3 = KMeans(n_clusters = k)
    model3.fit(X_train_standardScale)
    inertia.append(model3.inertia_)
    
inertia

plt.plot(l, inertia, '-o')
plt.xlabel("number of cluster K")
plt.ylabel("inertia")
plt.xticks(l)
plt.show()

l = range(1,31)
inertia = []
for k in l:
    model3 = KMeans(n_clusters = k)
    model3.fit(X_train_standardScale)
    inertia.append(model3.inertia_)
    
inertia

plt.plot(l, inertia, '-o')
plt.xlabel("number of cluster K")
plt.ylabel("inertia")
plt.xticks(l)
plt.show()

## 정규화(normalization) ##
model4 = KMeans(n_clusters = 4)

import copy
X = copy.deepcopy(mergeDF)
X.columns


from sklearn.preprocessing import MinMaxScaler
X_train = X.iloc[:,2:]
type(scale(X_train))
X_train_normalScale = MinMaxScaler().fit_transform(X_train)

model4.fit(X_train_normalScale)
model4.labels_
model4.cluster_centers_

colormatp = np.array(['red','blue','green','black'])
plt.scatter(mergeDF.iloc[:,2], mergeDF.iloc[:,3],
            c = colormatp[model4.labels_], s = 2)

centers = pd.DataFrame(model4.cluster_centers_)
plt.scatter(centers.iloc[:,0], centers.iloc[:,1], s = 50,
            marker = 'D', c='g')
plt.show()

fig = plt.figure()
axes3d = plt.axes(projection = '3d')
axes3d.scatter(X.iloc[:,2], X.iloc[:,3], X.iloc[:,4],
               c = colormatp[model4.labels_], s = 2)
plt.show()

#응집도 
model4.inertia_

l = range(1,11)
inertia = []
for k in l:
    model4 = KMeans(n_clusters = k)
    model4.fit(X_train_normalScale)
    inertia.append(model4.inertia_)
    
inertia

plt.plot(l, inertia, '-o')
plt.xlabel("number of cluster K")
plt.ylabel("inertia")
plt.xticks(l)
plt.show() 

########################################################################
#goty 표시하기 전에 2018년도 이전과 2019년도 작품을 분리하자 
mergeA = mergeDF[mergeDF['year_x'] <= 2018]
mergeB = mergeDF[mergeDF['year_x'] == 2019]
len(mergeA)
len(mergeB)

mergeB[mergeB['metascore'] >= 90]

mergeA['goty'] = [2 if i in gotyA else 1 if i in gotyB else 0 for i in mergeA['name']]
mergeA.sort_values(by = 'metascore')
mergeA[mergeA['goty'] >= 1]

mergeA.info()
mergeB.info()

mergeA.to_csv("C:/WorkSpace/Python_Space/Project/mergeA.csv")
mergeB.to_csv("C:/WorkSpace/Python_Space/Project/mergeB.csv")

#불러오기 
mergeA = pd.read_csv("C:/WorkSpace/Python_Space/Project/mergeA.csv")
mergeA = mergeA.iloc[:,1:]
mergeB = pd.read_csv("C:/WorkSpace/Python_Space/Project/mergeB.csv")
mergeB = mergeB.iloc[:,1:]

mergeA.info()#2359개 row
mergeB.info()#326개 row 
### knn
from sklearn.neighbors import KNeighborsClassifier
#train set과 test set을 나누자 
from sklearn.model_selection import train_test_split
import collections
from sklearn.metrics import classification_report, confusion_matrix

def knnClass(X,Y,Z):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)
    clf = KNeighborsClassifier(n_neighbors = 3)
    #훈련시키기 
    clf.fit(X_train, Y_train)
    print("정답률 점수 :",clf.score(X_test, Y_test))
    #이제 새로운 점을 넣어보자
    pred_z = clf.predict(Z)
    print("2019 goty 예상 :",collections.Counter(pred_z))
    pred_X = clf.predict(X_test)
    print("예측 갯수 : ",collections.Counter(pred_X))
    print("실재 갯수 : ",collections.Counter(Y_test))
    print(confusion_matrix(Y_test,pred_X))
    goty_suc = 0
    goty_sum = 0
    for i in [1,2]:
        goty_suc += confusion_matrix(Y_test,pred_X)[i][i]
        goty_sum += collections.Counter(Y_test)[i]
    pro = goty_suc / goty_sum
    print("원하는 부분의 정답률 (%):",pro * 100)
    print(classification_report(Y_test, pred_X))
      
#n : 반복횟수
#m : 이웃 수 
def knnClass2(X,Y,Z,n,m):
    tempDF = DataFrame(columns = ['n','score','goty score'])
    for j in range(0,n):
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)
        clf = KNeighborsClassifier(n_neighbors = m)
        #훈련시키기 
        clf.fit(X_train, Y_train)
        tempDF.at[j,'score'] = clf.score(X_test, Y_test)
        pred_X = clf.predict(X_test)
        goty_suc = 0
        goty_sum = 0
        for k in [1,2]:
            goty_suc += confusion_matrix(Y_test,pred_X)[k][k]
            goty_sum += collections.Counter(Y_test)[k]
        pro = goty_suc / goty_sum
        tempDF.at[j,'goty score'] = pro * 100
        tempDF.at[j,'n'] = j + 1
    print("평균 정확도 : {}".format(tempDF['score'].mean()))
    print("평균 goty score (%): {}".format(tempDF['goty score'].mean()))
    plt.plot(tempDF['n'],tempDF['goty score'])
    plt.show()


#KNN도 2가지 방법으로 스케일링을 하자 
## 표준화(standardization) ##
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(mergeA.iloc[:,2:5])
#분류기준 
Y = mergeA['goty']
#이제 새로운 점을 넣어보자 #[metascore, userscore, openscore](0 ~ 100)
Z = StandardScaler().fit_transform(mergeB.iloc[:,2:5])

standardScaler = StandardScaler()
standardScaler.fit(mergeA.iloc[:,2:5])
X = standardScaler.transform(mergeA.iloc[:,2:5])
Y = mergeA['goty']
Z = standardScaler.transform(mergeB.iloc[:,2:5])

knnClass(X,Y,Z)
#정답률 점수 : 0.9661016949152542
#2019 goty 예상 : Counter({0: 325, 1: 1})
#예측 갯수 :  Counter({0: 705, 1: 2, 2: 1})
#실재 갯수 :  Counter({0: 686, 1: 16, 2: 6})
#[[684   2   0]
# [ 15   0   1]
# [  6   0   0]]
#원하는 부분의 정답률 (%): 0.0
#              precision    recall  f1-score   support
#
#           0       0.97      1.00      0.98       686
#           1       0.00      0.00      0.00        16
#           2       0.00      0.00      0.00         6
#
#    accuracy                           0.97       708
#   macro avg       0.32      0.33      0.33       708
#weighted avg       0.94      0.97      0.95       708

#Counter({0: 325, 1: 1})
#1이 1개뿐이고 2는 없다.... 결과가 납득되지 않는다.

#[[684   2   0]
# [ 15   0   1]
# [  6   0   0]]
#정답률이 높게 나왔던게 우리가 원하는 것과 거리가 멀다.
#goty = 1,2 예측이 잘 안된다.

for i in range(3,30,2):
    print("neighbor 갯수 : {}".format(i))
    knnClass2(X,Y,Z,30,i)

from sklearn.model_selection import cross_val_score
clf = KNeighborsClassifier(n_neighbors = 3)
scores = cross_val_score(clf, X, Y, cv = 5)
print(scores)

colormatp = np.array(['yellow','blue','red'])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
axes3d = plt.axes(projection = '3d')
axes3d.scatter(mergeA.iloc[:,2], mergeA.iloc[:,3], mergeA.iloc[:,4],
               c = colormatp[Y], s = 2)
plt.show()

#보니 goty를 받지 못한 작품이 절대적으로 대다수이므로
#근접한 작품중에서 goty 0 인 작품이 더 많을 것이다. 
#그러므로 goty 2 없고, goty 1이 하나만 나온다. 


#K의 갯수를 다르게 해 보자
def knnK(X,Y,Z):
    tempDF = DataFrame(columns = ['k','score','goty score'])
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)
    j = 0
    for k in range(3,50,2):
        goty_sum = 0
        goty_suc = 0
        tempDF.at[j,'k'] = k
        #근접 점의 갯수
        clf = KNeighborsClassifier(n_neighbors = k)
        #훈련시키기 
        clf.fit(X_train, Y_train)
        pred_X = clf.predict(X_test)
        tempDF.at[j,'score'] = clf.score(X_test, Y_test)
        for i in [1,2]:
            goty_suc += confusion_matrix(Y_test,pred_X)[i][i]
            goty_sum += collections.Counter(Y_test)[i]

        tempDF.at[j,'goty score'] = goty_suc / goty_sum
        j += 1

    print(tempDF)
    plt.plot(tempDF['k'],tempDF['goty score'])
    plt.xlabel("k 값")
    plt.ylabel("goty 1,2 일때 맞춘 확률")
    plt.show()

knnK(X,Y,Z)
#집 다시 

#정규화 해서 해 보자 
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
X = minmax.fit_transform(mergeA.iloc[:,2:5])
Y = mergeA['goty']
Z = minmax.transform(mergeB.iloc[:,2:5])

knnClass(X,Y,Z)
#정답률 점수 : 0.9774011299435028
#2019 goty 예상 : Counter({0: 317, 1: 6, 2: 3})
#예측 갯수 :  Counter({0: 702, 1: 5, 2: 1})
#실재 갯수 :  Counter({0: 691, 1: 14, 2: 3})
#[[689   1   1]
# [ 11   3   0]
# [  2   1   0]]
#원하는 부분의 정답률 (%): 17.647058823529413
#              precision    recall  f1-score   support
#
#           0       0.98      1.00      0.99       691
#           1       0.60      0.21      0.32        14
#           2       0.00      0.00      0.00         3
#
#    accuracy                           0.98       708
#   macro avg       0.53      0.40      0.44       708
#weighted avg       0.97      0.98      0.97       708

knnK(X,Y,Z)

#이번에는 다르게 장르들도 포함해서 실행해보자 
#먼저 정규화를 해서 실행하자 
standardScaler = StandardScaler()
X = standardScaler.fit_transform(mergeA.iloc[:,2:-1])
Y = mergeA['goty']
Z = standardScaler.transform(mergeB.iloc[:,2:])
knnClass(X,Y,Z)
#정답률 점수 : 0.9646892655367232
#2019 goty 예상 : Counter({0: 322, 1: 3, 2: 1})
#예측 갯수 :  Counter({0: 700, 1: 7, 2: 1})
#실재 갯수 :  Counter({0: 689, 1: 16, 2: 3})
#[[683   6   0]
# [ 15   0   1]
# [  2   1   0]]
#원하는 부분의 정답률 (%): 0.0
#              precision    recall  f1-score   support
#
#           0       0.98      0.99      0.98       689
#           1       0.00      0.00      0.00        16
#           2       0.00      0.00      0.00         3
#
#    accuracy                           0.96       708
#   macro avg       0.33      0.33      0.33       708
#weighted avg       0.95      0.96      0.96       708

knnK(X,Y,Z)

#이번에는 표준화를 해서 실행하자
X = MinMaxScaler().fit_transform(mergeA.iloc[:,2:-1])
Y = mergeA['goty']
Z = MinMaxScaler().fit_transform(mergeB.iloc[:,2:])
knnClass(X,Y,Z)
#정답률 점수 : 0.9646892655367232
#2019 goty 예상 : Counter({0: 320, 1: 5, 2: 1})
#예측 갯수 :  Counter({0: 700, 1: 8})
#실재 갯수 :  Counter({0: 688, 1: 12, 2: 8})
#[[682   6   0]
# [ 11   1   0]
# [  7   1   0]]
#원하는 부분의 정답률 (%): 5.0
#              precision    recall  f1-score   support
#
#           0       0.97      0.99      0.98       688
#           1       0.12      0.08      0.10        12
#           2       0.00      0.00      0.00         8
#
#    accuracy                           0.96       708
#   macro avg       0.37      0.36      0.36       708
#weighted avg       0.95      0.96      0.96       708

knnK(X,Y,Z)

#####################################################################
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
label = mergeA['goty']
log.fit(mergeA.iloc[:,2:5],label)

pred2 = log.predict(mergeB.iloc[:,2:5])
collections.Counter(pred2)
#Counter({0: 326}) ...

def logisticRegression(X,Y,Z):
    log = LogisticRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)
    log.fit(X_train, Y_train)
    pred_X = log.predict(X_test)
    print("정답률 점수 :",clf.score(X_test, Y_test))
    print("예측 갯수 :",collections.Counter(pred_X))
    print("실제 갯수 :",collections.Counter(Y_test))
    print(confusion_matrix(Y_test,pred_X))
    print(classification_report(Y_test, pred_X))
    goty_suc = 0
    goty_sum = 0
    for i in [1,2]:
        goty_suc += confusion_matrix(Y_test,pred_X)[i][i]
        goty_sum += collections.Counter(Y_test)[i]
    pro = goty_suc / goty_sum
    print("원하는 부분의 정답률 (%):",pro * 100)
    pred_X2 = log.predict(Z)
    print("2019 goty 예상 :",collections.Counter(pred_X2))
    

#train과 test를 나누어서 해 보자 
X = mergeA.iloc[:,2:5]
Y = mergeA['goty']
Z = mergeB.iloc[:,2:5]
logisticRegression(X,Y,Z) 
#예측 갯수 : Counter({0: 708})
#실제 갯수 : Counter({0: 685, 1: 16, 2: 7})
#[[685   0   0]
# [ 16   0   0]
# [  7   0   0]]
#              precision    recall  f1-score   support
#
#           0       0.97      1.00      0.98       685
#           1       0.00      0.00      0.00        16
#           2       0.00      0.00      0.00         7
#
#    accuracy                           0.97       708
#   macro avg       0.32      0.33      0.33       708
#weighted avg       0.94      0.97      0.95       708
#
#원하는 부분의 정답률 (%): 0.0
#2019 goty 예상 : Counter({0: 326})

#이번에는 장르를 포함해서 해 보자 
X = mergeA.iloc[:,2:-1]
Y = mergeA['goty']
Z = mergeB.iloc[:,2:]
logisticRegression(X,Y,Z) 
#예측 갯수 : Counter({0: 708})
#실제 갯수 : Counter({0: 690, 1: 14, 2: 4})
#[[690   0   0]
# [ 14   0   0]
# [  4   0   0]]
#              precision    recall  f1-score   support
#
#           0       0.97      1.00      0.99       690
#           1       0.00      0.00      0.00        14
#           2       0.00      0.00      0.00         4
#
#    accuracy                           0.97       708
#   macro avg       0.32      0.33      0.33       708
#weighted avg       0.95      0.97      0.96       708
#
#원하는 부분의 정답률 (%): 0.0
#2019 goty 예상 : Counter({0: 326})

#표준화 스케일링을 해 보자 
X = StandardScaler().fit_transform(mergeA.iloc[:,2:5])
Y = mergeA['goty']
Z = StandardScaler().fit_transform(mergeB.iloc[:,2:5])
logisticRegression(X,Y,Z) 
#예측 갯수 : Counter({0: 708})
#실제 갯수 : Counter({0: 680, 1: 23, 2: 5})
#[[680   0   0]
# [ 23   0   0]
# [  5   0   0]]
#              precision    recall  f1-score   support
#
#           0       0.96      1.00      0.98       680
#           1       0.00      0.00      0.00        23
#           2       0.00      0.00      0.00         5
#
#    accuracy                           0.96       708
#   macro avg       0.32      0.33      0.33       708
#weighted avg       0.92      0.96      0.94       708
#
#원하는 부분의 정답률 (%): 0.0
#2019 goty 예상 : Counter({0: 326})

#정규화 스케일링을 해 보자 
X = MinMaxScaler().fit_transform(mergeA.iloc[:,2:5])
Y = mergeA['goty']
Z = MinMaxScaler().fit_transform(mergeB.iloc[:,2:5])
logisticRegression(X,Y,Z) 
#예측 갯수 : Counter({0: 708})
#실제 갯수 : Counter({0: 689, 1: 12, 2: 7})
#[[689   0   0]
# [ 12   0   0]
# [  7   0   0]]
#              precision    recall  f1-score   support
#
#           0       0.97      1.00      0.99       689
#           1       0.00      0.00      0.00        12
#           2       0.00      0.00      0.00         7
#
#    accuracy                           0.97       708
#   macro avg       0.32      0.33      0.33       708
#weighted avg       0.95      0.97      0.96       708
#
#원하는 부분의 정답률 (%): 0.0
#2019 goty 예상 : Counter({0: 326})

#장르를 포함해서 표준화 스케일링을 해보자 
X = StandardScaler().fit_transform(mergeA.iloc[:,2:-1])
Y = mergeA['goty']
Z = StandardScaler().fit_transform(mergeB.iloc[:,2:])
logisticRegression(X,Y,Z) 
#예측 갯수 : Counter({0: 703, 1: 5})
#실제 갯수 : Counter({0: 688, 1: 13, 2: 7})
#[[686   2   0]
# [ 12   1   0]
# [  5   2   0]]
#              precision    recall  f1-score   support
#
#           0       0.98      1.00      0.99       688
#           1       0.20      0.08      0.11        13
#           2       0.00      0.00      0.00         7
#
#    accuracy                           0.97       708
#   macro avg       0.39      0.36      0.37       708
#weighted avg       0.95      0.97      0.96       708
#
#원하는 부분의 정답률 (%): 5.0
#2019 goty 예상 : Counter({0: 326})

#장르를 포함해서 정규화 스케일링을 해보자 
X = MinMaxScaler().fit_transform(mergeA.iloc[:,2:-1])
Y = mergeA['goty']
Z = MinMaxScaler().fit_transform(mergeB.iloc[:,2:])
logisticRegression(X,Y,Z) 
#예측 갯수 : Counter({0: 708})
#실제 갯수 : Counter({0: 691, 1: 14, 2: 3})
#[[691   0   0]
# [ 14   0   0]
# [  3   0   0]]
#              precision    recall  f1-score   support
#
#           0       0.98      1.00      0.99       691
#           1       0.00      0.00      0.00        14
#           2       0.00      0.00      0.00         3
#
#    accuracy                           0.98       708
#   macro avg       0.33      0.33      0.33       708
#weighted avg       0.95      0.98      0.96       708
#
#원하는 부분의 정답률 (%): 0.0
#2019 goty 예상 : Counter({0: 326})

#####################################################################
from sklearn.tree import DecisionTreeClassifier


def DTC(X,Y,Z,cri,n):
    modelTree = DecisionTreeClassifier(criterion = cri, max_depth = n)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)

    modelTree.fit(X_train, Y_train)
    predTree = modelTree.predict(X_test)
    
    print("정답률 점수 : ",modelTree.score(X_test, Y_test))
    print("예측 갯수 :",collections.Counter(predTree))
    print("실제 갯수 :",collections.Counter(Y_test))
    print(confusion_matrix(Y_test,predTree))
    print(classification_report(Y_test, predTree))
    tempDF = DataFrame({'feature':X.columns,
                        'importances':modelTree.feature_importances_})
    print(tempDF)
    goty_suc = 0
    goty_sum = 0
    for i in [1,2]:
        goty_suc += confusion_matrix(Y_test,predTree)[i][i]
        goty_sum += collections.Counter(Y_test)[i]
    pro = goty_suc / goty_sum
    print("원하는 부분의 정답률 (%):",pro * 100)
    pred_z = modelTree.predict(Z)
    print("2019 goty 예측 : ",collections.Counter(pred_z))
    
X = mergeA.iloc[:,2:5]
Y = mergeA['goty']
Z = mergeB.iloc[:,2:5]
DTC(X,Y,Z,'entropy',5)
DTC(X,Y,Z,'gini',5)

#그림을 그려보자 
import pydotplus
import graphviz

from sklearn.tree import export_graphviz
from IPython.display import Image

col = ['metascore', 'userscore', 'openscore']

dot_data = export_graphviz(modelTree, out_file=None,
                           feature_names= mergeA.iloc[:,2:5].columns,
                           filled=True,rounded=True,
                           special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
plt.show()

#max depth 값에 따라서 달라질까?
def DTCMD(X,Y,Z,cri):
    tempDF = DataFrame(columns = ['max depth','score','goty score'])
    j = 0
    for i in range(3,50):
        tempDF.at[j,'max depth'] = i
        modelTree = DecisionTreeClassifier(criterion = cri, max_depth = i)
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)
        modelTree.fit(X_train, Y_train)
        predTree = modelTree.predict(X_test)

        tempDF.at[j,'score'] = modelTree.score(X_test, Y_test)

        goty_suc = 0
        goty_sum = 0
        for i in [1,2]:
            goty_suc += confusion_matrix(Y_test,predTree)[i][i]
            goty_sum += collections.Counter(Y_test)[i]

        tempDF.at[j,'goty score'] = goty_suc / goty_sum
        j += 1
    
    print(tempDF)
    print("평균 정확도 : {}".format(tempDF['score'].mean()))
    print("평균 goty score (%): {}".format(tempDF['goty score'].mean()))
    plt.plot(tempDF['max depth'], tempDF['goty score'])
    plt.show()
    
DTCMD(X,Y,Z,'entropy')
DTCMD(X,Y,Z,'gini')


#장르를 넣고 해 보자 
X = mergeA.iloc[:,2:-1]
Y = mergeA['goty']
Z = mergeB.iloc[:,2:]
DTC(X,Y,Z,'entropy',5)
DTC(X,Y,Z,'gini',5)

DTCMD(X,Y,Z,'entropy')
DTCMD(X,Y,Z,'gini')

##########################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def RFC(X,Y,Z,cri,esti):
    model = RandomForestClassifier(n_estimators = esti,
                                   oob_score = True,#default = False
                                   criterion = cri,
                                   random_state = 0)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)

    model.fit(X_train, Y_train)
    predTree = model.predict(X_test)
    
    print("정답률 점수 : ",model.score(X_test, Y_test))
    print("예측 갯수 :",collections.Counter(predTree))
    print("실제 갯수 :",collections.Counter(Y_test))
    print(confusion_matrix(Y_test,predTree))
    print(classification_report(Y_test, predTree))
    tempDF = DataFrame({'feature':X.columns,
                        'importances':model.feature_importances_})
    print(tempDF)
    goty_suc = 0
    goty_sum = 0
    for i in [1,2]:
        goty_suc += confusion_matrix(Y_test,predTree)[i][i]
        goty_sum += collections.Counter(Y_test)[i]
    pro = goty_suc / goty_sum
    print("원하는 부분의 정답률 (%):",pro * 100)
    pred_z = model.predict(Z)
    print("2019 goty 예측 : ",collections.Counter(pred_z))
   
X = mergeA.iloc[:,2:5]
Y = mergeA['goty']
Z = mergeB.iloc[:,2:5]
RFC(X,Y,Z,'entropy',10)
RFC(X,Y,Z,'gini',10)

#장르를 넣고 해 보자
X = mergeA.iloc[:,2:-1]
Y = mergeA['goty']
Z = mergeB.iloc[:,2:]
RFC(X,Y,Z,'entropy',10)
RFC(X,Y,Z,'gini',10)
