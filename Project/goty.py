import pandas as pd
import numpy as np 

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


mergeDF = pd.read_csv("C:/WorkSpace/PythonSpace/Python_Space/Project/merge.csv")
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
mergeA = pd.read_csv("C:/WorkSpace/PythonSpace/Python_Space/Project/mergeA.csv")
mergeA = mergeA.iloc[:,1:]
mergeB = pd.read_csv("C:/WorkSpace/PythonSpace/Python_Space/Project/mergeB.csv")
mergeB = mergeB.iloc[:,1:]

mergeA.info()#2359개 row
mergeB.info()#326개 row 
### knn
from sklearn.neighbors import KNeighborsClassifier

#KNN도 2가지 방법으로 스케일링을 하자 
## 표준화(standardization) ##
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(mergeA.iloc[:,2:5])
X

#훈련 데이터 셋
X = np.array(X)
#분류기준 
label = mergeA['goty']

#train set과 test set을 나누자 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,label,test_size = 0.3)

#근접 점의 갯수
np.sqrt(2359)#48.569537778323564
clf = KNeighborsClassifier(n_neighbors = 3)
#훈련시키기 
clf.fit(X_train, Y_train)

clf.score(X_test, Y_test)
#0.9682203389830508

#이제 새로운 점을 넣어보자 #[metascore, userscore, openscore](0 ~ 100)
Z = StandardScaler().fit_transform(mergeB.iloc[:,2:5])
pred_z = clf.predict(Z)

import collections
type(pred_z)
len(pred_z)
collections.Counter(pred_z)
#Counter({0: 325, 1: 1})
#1이 1개뿐이고 2는 없다.... 결과가 납득되지 않는다.

pred_X = clf.predict(X_test)
collections.Counter(pred_X)#Counter({0: 702, 1: 6}) 예측 갯수
collections.Counter(Y_test)#Counter({0: 690, 1: 12, 2: 6}) 실제 갯수 

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test,pred_X))
confusion_matrix(Y_test,pred_X)[1][1]
collections.Counter(Y_test)[2]
"""
[[685   5   0]
 [ 11   1   0]
 [  6   0   0]]"""
#정답률이 높게 나왔던게 우리가 원하는 것과 거리가 멀다.
#goty = 1,2 예측이 잘 안된다.

print(classification_report(Y_test, pred_X))
""" 
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       690
           1       0.17      0.08      0.11        12
           2       0.00      0.00      0.00         6

   micro avg       0.97      0.97      0.97       708
   macro avg       0.38      0.36      0.37       708
weighted avg       0.95      0.97      0.96       708 """

colormatp = np.array(['yellow','blue','red'])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
axes3d = plt.axes(projection = '3d')
axes3d.scatter(mergeA.iloc[:,2], mergeA.iloc[:,3], mergeA.iloc[:,4],
               c = colormatp[label], s = 2)
plt.show()

 
#보니 goty를 받지 못한 작품이 절대적으로 대다수이므로
#근접한 작품중에서 goty 0 인 작품이 더 많을 것이다. 
#그러므로 goty 2 없고, goty 1이 하나만 나온다. 

tempDF = DataFrame(columns = ['k','score','goty score'])
#K의 갯수를 다르게 해 보자
j = 0
for k in range(3,50,2):
    goty_sum = 0
    goty_suc = 0
    tempDF.at[j,'k'] = k
    #근접 점의 갯수
    clf = KNeighborsClassifier(n_neighbors = k)
    #훈련시키기 
    clf.fit(X_train, Y_train)

    tempDF.at[j,'score'] = clf.score(X_test, Y_test)
    for i in [1,2]:
        goty_suc += confusion_matrix(Y_test,pred_X)[i][i]
        goty_sum += collections.Counter(Y_test)[i]

    tempDF.at[j,'goty score'] = goty_suc / goty_sum
    j += 1

tempDF
"""
     k     score goty score
0    3  0.968927  0.0555556
1    5  0.970339  0.0555556
2    7  0.971751  0.0555556
3    9  0.974576  0.0555556
4   11  0.974576  0.0555556
5   13  0.974576  0.0555556
6   15  0.974576  0.0555556
7   17  0.974576  0.0555556
8   19  0.974576  0.0555556
9   21  0.974576  0.0555556
10  23  0.974576  0.0555556
11  25  0.974576  0.0555556
12  27  0.974576  0.0555556
13  29  0.974576  0.0555556
14  31  0.974576  0.0555556
15  33  0.974576  0.0555556
16  35  0.974576  0.0555556
17  37  0.974576  0.0555556
18  39  0.974576  0.0555556
19  41  0.974576  0.0555556
20  43  0.974576  0.0555556
21  45  0.974576  0.0555556
22  47  0.974576  0.0555556
23  49  0.974576  0.0555556"""

plt.plot(tempDF['k'],tempDF['goty score'])
plt.xlabel("k 값")
plt.ylabel("goty 1,2 일때 맞춘 확률")
plt.show()

#이번에는 다르게 장르들도 포함해서 실행해보자 
X = StandardScaler().fit_transform(mergeA.iloc[:,2:-1])
X

#훈련 데이터 셋
X = np.array(X)
#분류기준 
label = mergeA['goty']

#근접 점의 갯수
clf2 = KNeighborsClassifier(n_neighbors = 3)

X_train, X_test, Y_train, Y_test = train_test_split(X,label,test_size = 0.3)
#훈련시키기 
clf2.fit(X_train,Y_train)

clf2.score(X_test, Y_test)
#0.9646892655367232

Z = StandardScaler().fit_transform(mergeB.iloc[:,2:])
pred_z = clf2.predict(Z)
collections.Counter(pred_z)
#Counter({0: 322, 2: 2, 1: 2})

pred_X = clf2.predict(X_test)
collections.Counter(pred_X)#Counter({0: 694, 1: 12, 2: 2}) 예측 갯수
collections.Counter(Y_test)#Counter({0: 689, 1: 14, 2: 5}) 실제 갯수 

print(confusion_matrix(Y_test,pred_X))
"""
[[680   9   0]
 [ 11   2   1]
 [  3   1   1]]"""
print(classification_report(Y_test, pred_X))
"""
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       689
           1       0.17      0.14      0.15        14
           2       0.50      0.20      0.29         5

   micro avg       0.96      0.96      0.96       708
   macro avg       0.55      0.44      0.47       708
weighted avg       0.96      0.96      0.96       708"""

tempDF2 = DataFrame(columns = ['k','score','goty score'])
#K의 갯수를 다르게 해 보자
j = 0
for k in range(3,50,2):
    goty_sum = 0
    goty_suc = 0
    tempDF2.at[j,'k'] = k
    #근접 점의 갯수
    clf2 = KNeighborsClassifier(n_neighbors = k)
    #훈련시키기 
    clf2.fit(X_train, Y_train)

    tempDF2.at[j,'score'] = clf2.score(X_test, Y_test)
    for i in [1,2]:
        goty_suc += confusion_matrix(Y_test,pred_X)[i][i]
        goty_sum += collections.Counter(Y_test)[i]

    tempDF2.at[j,'goty score'] = goty_suc / goty_sum
    j += 1

tempDF2
"""
     k     score goty score
0    3  0.964689   0.157895
1    5  0.968927   0.157895
2    7  0.968927   0.157895
3    9  0.968927   0.157895
4   11  0.968927   0.157895
5   13  0.973164   0.157895
6   15  0.973164   0.157895
7   17  0.973164   0.157895
8   19  0.973164   0.157895
9   21  0.973164   0.157895
10  23  0.973164   0.157895
11  25  0.973164   0.157895
12  27  0.973164   0.157895
13  29  0.973164   0.157895
14  31  0.973164   0.157895
15  33  0.973164   0.157895
16  35  0.973164   0.157895
17  37  0.973164   0.157895
18  39  0.973164   0.157895
19  41  0.973164   0.157895
20  43  0.973164   0.157895
21  45  0.973164   0.157895
22  47  0.973164   0.157895
23  49  0.973164   0.157895"""

plt.plot(tempDF2['k'],tempDF2['goty score'])
plt.xlabel("k 값")
plt.ylabel("goty 1,2 일때 맞춘 확률")
plt.show()

#####################################################################
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(mergeA.iloc[:,2:5],label)

pred2 = log.predict(mergeB.iloc[:,2:5])
collections.Counter(pred2)

log2 = LogisticRegression()
log2.fit(mergeA.iloc[:,2:-1],label)

pred22 = log2.predict(mergeB.iloc[:,2:])
collections.Counter(pred22)

X = StandardScaler().fit_transform(mergeA.iloc[:,2:-])
Y = StandardScaler().fit_transform(mergeB.iloc[:,2:])

log3 = LogisticRegression()
log3.fit(X,label)

pred23 = log3.predict(Y)
collections.Counter(pred23)

#####################################################################
from sklearn.tree import DecisionTreeClassifier

modelTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)

X = StandardScaler().fit_transform(mergeA.iloc[:,2:5])
label = mergeA['goty']

X_train, X_test, Y_train, Y_test = train_test_split(X,label,test_size = 0.3)

modelTree.fit(X_train, Y_train)
predTree = modelTree.predict(X_test)

modelTree.score(X_test, Y_test)
#0.9759887005649718

collections.Counter(predTree)
#Counter({0: 700, 1: 6, 2: 2})

Z = StandardScaler().fit_transform(mergeB.iloc[:,2:5])
pred_z = modelTree.predict(Z)
collections.Counter(pred_z)
#Counter({0: 323, 1: 3})

pred_X = modelTree.predict(X_test)
collections.Counter(pred_X)#Counter({0: 700, 1: 6, 2: 2}) 예측 갯수
collections.Counter(Y_test)#Counter({0: 690, 1: 16, 2: 2}) 실제 갯수 

print(confusion_matrix(Y_test,pred_X))
"""
[[687   3   0]
 [ 12   3   1]
 [  1   0   1]]"""

"""
#그림을 그려보자 
import pydotplus
import graphviz

from sklearn.tree import export_graphviz
from IPython.display import Image

mergeA.iloc[:,2:-1].columns

dot_data = export_graphviz(modelTree, out_file=None,
                           feature_names= mergeA.iloc[:,2:-1].columns,
                           filled=True,rounded=True,
                           special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
plt.show()"""

#max depth 값에 따라서 달라질까?
tempDF3 = DataFrame(columns = ['max depth','score','goty score'])
j = 0
for i in range(3,30):
    tempDF3.at[j,'max depth'] = i
    modelTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = i)
    modelTree.fit(X_train, Y_train)
    predTree = modelTree.predict(X_test)

    tempDF3.at[j,'score'] = modelTree.score(X_test, Y_test)

    for i in [1,2]:
        goty_suc += confusion_matrix(Y_test,predTree)[i][i]
        goty_sum += collections.Counter(Y_test)[i]

    tempDF3.at[j,'goty score'] = goty_suc / goty_sum
    j += 1

tempDF3
plt.plot(tempDF3['max depth'], tempDF3['goty score'])
plt.show()

