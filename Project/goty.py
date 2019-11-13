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

colormatp = np.array(['red','blue','green','black'])
plt.scatter(mergeDF.iloc[:,2], mergeDF.iloc[:,3],
            c = colormatp[model.labels_], s = 2)

centers = pd.DataFrame(model.cluster_centers_)
plt.scatter(centers.iloc[:,0], centers.iloc[:,1], s = 50,
            marker = 'D', c='g')
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

## 정규화(normalization) ##
model2 = KMeans(n_clusters = 4)

import copy
X = copy.deepcopy(mergeDF)
X.columns


from sklearn.preprocessing import scale
X_train = X.iloc[:,2:5]
type(scale(X_train))
X_train_normalScale = scale(X_train)

model2.fit(X_train_normalScale)
model2.labels_
model2.cluster_centers_

colormatp = np.array(['red','blue','green','black'])
plt.scatter(mergeDF.iloc[:,2], mergeDF.iloc[:,3],
            c = colormatp[model2.labels_], s = 2)

centers = pd.DataFrame(model.cluster_centers_)
plt.scatter(centers.iloc[:,0], centers.iloc[:,1], s = 50,
            marker = 'D', c='g')
plt.show()

#응집도 
model2.inertia_

l = range(1,11)
inertia = []
for k in l:
    model2 = KMeans(n_clusters = k)
    model2.fit(X_train_standardScale)
    inertia.append(model2.inertia_)
    
inertia

plt.plot(l, inertia, '-o')
plt.xlabel("number of cluster K")
plt.ylabel("inertia")
plt.xticks(l)

#두 스케일로 그룹이 달라지는가? 근데 각 위치별로 숫자가 일치하는건 아닌데
#... 

(model.labels_ == model2.labels_).sum()

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
### knn
from sklearn.neighbors import KNeighborsClassifier

#KNN도 2가지 방법으로 스케일링을 하자 
## 표준화(standardization) ##
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(mergeA.iloc[:,2:5])
X

#훈련 데이터 셋
x_train = np.array(X)
#분류기준 
label = mergeA['goty']

#근접 점의 갯수
clf = KNeighborsClassifier(n_neighbors = 3)
#훈련시키기 
clf.fit(x_train,label)
#이제 새로운 점을 넣어보자 #[metascore, userscore, openscore](0 ~ 100)
clf.predict(np.array([[95,95,95]]))[0]

Y = StandardScaler().fit_transform(mergeB.iloc[:,2:5])
pred = clf.predict(Y)

import collections
type(pred)
len(pred)
collections.Counter(pred)

j = 0
for i in pred:
    print(i)
    if i > 0:
        print(mergeB.iloc[j,:])
    j +=1
    
colormatp = np.array(['yellow','blue','red'])
plt.scatter(mergeA.iloc[:,1], mergeA.iloc[:,3],
            c = colormatp[label], s = 2)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
axes3d = plt.axes(projection = '3d')
axes3d.scatter(mergeA.iloc[:,2], mergeA.iloc[:,3], mergeA.iloc[:,4],
               c = colormatp[label], s = 2)

 
#보니 goty를 받지 못한 작품이 절대적으로 대다수이므로
#근접한 작품중에서 goty 0 인 작품이 더 많을 것이다. 
#그러므로 goty 2 없고, goty 1이 하나만 나온다. 

#이번에는 다르게 장르들도 포함해서 실행해보자 
X = StandardScaler().fit_transform(mergeA.iloc[:,2:-1])
X

#훈련 데이터 셋
x_train = np.array(X)
#분류기준 
label = mergeA['goty']

#근접 점의 갯수
clf = KNeighborsClassifier(n_neighbors = 3)
#훈련시키기 
clf.fit(x_train,label)

Y = StandardScaler().fit_transform(mergeB.iloc[:,2:])
pred = clf.predict(Y)
collections.Counter(pred)

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

modelTree.fit(mergeA.iloc[:,2:-1],label)
predTree = modelTree.predict(mergeB.iloc[:,2:])

collections.Counter(predTree)

#Scale한거 넣어보자 
modelTree2 = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)

X = StandardScaler().fit_transform(mergeA.iloc[:,2:-1])
Y = StandardScaler().fit_transform(mergeB.iloc[:,2:])

modelTree2.fit(X,label)
predTree2 = modelTree.predict(Y)

collections.Counter(predTree2)


#그림을 그려보자 
import pydotplus
import graphviz

from sklearn.tree import export_graphviz
from IPython.display import Image

mergeA.iloc[:,2:-1].columns

dot_data = export_graphviz(modelTree2, out_file=None,
                           feature_names= mergeA.iloc[:,2:-1].columns,
                           filled=True,rounded=True,
                           special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())