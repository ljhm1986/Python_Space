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
count = 0
for i in metascoreDF['name']:
    if i in gotyA:
        count += 1
print(count)#39

count = 0
for i in opencriticDF['name']:
    if i in gotyA:
        count += 1
print(count)#15

count = 0
for i in metascoreDF['name']:
    if i in gotyB:
        count += 1
print(count)#153

count = 0
for i in opencriticDF['name']:
    if i in gotyB:
        count += 1
print(count)#53

metascoreDF[metascoreDF['name'].isin(gotyA)]


mergeDF = pd.read_csv("C:/WorkSpace/PythonSpace/Python_Space/Project/merge.csv")
mergeDF['name']

mergeDF['goty'] = [2 if i in gotyA else 1 if i in gotyB else 0 for i in mergeDF['name']]
mergeDF.sort_values(by = 'metascore')
mergeDF[mergeDF['goty'] >= 1]

mergeDF = mergeDF[['name','metascore','userscore','openscore','year_x','goty']]
mergeDF['userscore'] = mergeDF['userscore']*10

#k-mean 을 이용해서 여러 그룹으로 나누어 보자 
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 4)
model.fit(mergeDF.iloc[:,1:4])
model.labels_
model.cluster_centers_

#응집도 
model.inertia_

colormatp = np.array(['red','blue','green','black'])
plt.scatter(mergeDF.iloc[:,2], mergeDF.iloc[:,3],
            c = colormatp[model.labels_], s = 2)

centers = pd.DataFrame(model.cluster_centers_)
plt.scatter(centers.iloc[:,0], centers.iloc[:,1], s = 50,
            marker = 'D', c='g')
plt.show()

### knn
from sklearn.neighbors import KNeighborsClassifier
#훈련 데이터 셋
x_train = np.array(mergeDF.iloc[:,1:4])
#분류기준 
label = mergeDF['goty']

#근접 점의 갯수
clf = KNeighborsClassifier(n_neighbors = 3)
#훈련시키기 
clf.fit(x_train,label)
#이제 새로운 점을 넣어보자 #[metascore, userscore, openscore](0 ~ 100)
clf.predict(np.array([[95,95,95]]))[0]

merge2019 = pd.read_csv("C:/WorkSpace/PythonSpace/Python_Space/Project/merge2019.csv")
merge2019 = merge2019[['name','metascore','userscore','openscore','year_y']]
merge2019['userscore'] = merge2019['userscore']*10

clf.predict(merge2019.iloc[:,1:4])