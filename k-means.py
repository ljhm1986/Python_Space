# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:00:28 2019

@author: stu12
"""

#10/29
#k-means
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#한글이 있으니 UTF-8로 저장한다. 
academy = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\academy.csv")

academy
#영어, 수학 점수를 기준으로 4개 그룹으로 구분해보자 
model = KMeans(n_clusters = 4)
model.fit(academy.iloc[:,3:5])
model.labels_
model.cluster_centers_

import numpy as np
colormatp = np.array(['red','blue','green','yellow'])
plt.scatter(academy.iloc[:,3],academy.iloc[:,4],
            c = colormatp[model.labels_], s = 30)

centers = pd.DataFrame(model.cluster_centers_)
plt.scatter(centers.iloc[:,0], centers.iloc[:,1], s = 50,
            marker = 'D', c='g')
plt.show()

#응집도 
model.inertia_
#k값을 무엇으로 하는게 좋을까? 를 생각할 때 염두에 둘 수 있다.

l = range(1,11)
inertia = []
for k in l:
    model = KMeans(n_clusters = k)
    model.fit(academy.iloc[:,3:5])
    inertia.append(model.inertia_)
    
inertia

plt.plot(l, inertia, '-o')
plt.xlabel("number of cluster K")
plt.ylabel("inertia")
plt.xticks(l)

#inertia value는 군집화가 된 후에 각 중심점에서 군집의 데이터간의 거리를 
#합산한 것으로 군집의 응집도를 나타내는 값이다.
#이 값이 작을수록 응집도가 높게 군집화가 잘 되어있다고 평가할 수 있다. 
#기울기가 급격히 감소하는 지점이 아마도 좋은 k값일 것이다.

#Elbow method
#k값 변화에 따른 inertia 값이 변하지 않는 k를 선택하는 것이 합리적이다. 

########################################################################
#10/30#
#R책에 있는 내용을 python으로 해 보자 
import pandas as pd

snsData = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\snsdata.csv")
snsData
len(snsData)
snsData.info()#NaN은 갯수가 누락된다 
snsData.columns
#NaN이 있는지 보자 
snsData.isnull().any()
#gender와 age에 NaN이 있다. 갯수를 세어보자 
snsData.isnull().sum()

#성별의 숫자를 해아려보자 
from collections import Counter
Counter(snsData['gender'])
# Counter({'M': 5222, 'F': 22054, nan: 2724})
snsData['gender'].value_counts()
#F    22054
#M     5222
#Name: gender, dtype: int64
snsData['gender'].value_counts(dropna = False)
#F      22054
#M       5222
#NaN     2724
#Name: gender, dtype: int64
snsData['gender'].isnull().sum()

#요약 정보를 살펴보기(범주형 데이터) 
snsData['gender'].describe()
#count     27276
#unique        2
#top           F
#freq      22054
#Name: gender, dtype: object

#나이도 살펴보자 (수치형 데이터)
snsData['age'].describe()
#count    24914.000000
#mean        17.993950
#std          7.858054
#min          3.086000
#25%         16.312000
#50%         17.287000
#75%         18.259000
#max        106.927000
#Name: age, dtype: float64
#나이가 이상한게 보인다. 잘 못 입력되었을 수 있고 ..
snsData['age'].sum() / 24914
snsData['age'].count()#24914 , 원래 총 30000개 
snsData['age'] >= 20
snsData['age'] < 13

#고등학생의 나이가 아닌 수
Counter(snsData['age'] >= 20)
#Counter({False: 29616, True: 384})
Counter(snsData['age'] <13)
#Counter({False: 29947, True: 53})

#나이가 이상한 것을 출력해보자 
[i for i in snsData['age'] if (i >= 20) | (i < 13)]

#수정해보자 
snsData['age'] = [np.nan if (i >= 20) | (i < 13) else i 
                   for i in snsData['age']]

#내가 했던 수정방식 
for i in range(len(snsData)):
    if ((snsData['age'][i] >= 20) | (snsData['age'][i] < 13)):
        snsData['age'][i] = None
        
#다시 요약 통계량을 확인해보자, 통계량이 달라지는데
#평균, 표준편차를 구할때 row의 총 갯수로 나누는게 아니라
#NaN가 아닌 row의 갯수로만 나누기 때문이다.
snsData['age'].describe()
#count    24477.000000
#mean        17.252429
#std          1.157465
#min         13.027000
#25%         16.304000
#50%         17.265000
#75%         18.220000
#max         19.995000
#Name: age, dtype: float64        
Counter(snsData['age'].isnull())
#Counter({False: 24477, True: 5523})

#female만 해아리는 column을 만들어보자 
snsData['female'] = [1 if i == 'F' else 0 for i in snsData['gender']]
#map도 있긴 한데 NaN은 ???

#성별이 없는 것에 대해 따로 해아려 보자, 아래 둘의 결과는 같다.
snsData['no_gender'] = [0 if i in ['F','M'] else 1 for i in snsData['gender']]
snsData['no_gender'].value_counts()
snsData['no_gender'] = [1 if pd.isnull(i) else 0 for i in snsData['gender']]
snsData['no_gender'].value_counts()
#0    27276
#1     2724
#Name: no_gender, dtype: int64

#나이값이 없는것에 어떤 수를 넣어야 할까? 평균을 넣을까?
snsData['age'].mean()
np.mean(snsData['age'])
snsData['age'].mean(skipna = True)
snsData['age'].mean(skipna = False)

#졸업년도를 살펴보자 
snsData.info()
snsData['gradyear'].value_counts()
#2009    7500
#2008    7500
#2007    7500
#2006    7500
#Name: gradyear, dtype: int64
snsData[['age','gradyear']].mean()

#그럼 졸업년도를 기준으로 나이의 평균을 구해보자 
snsData['age'].groupby(snsData['gradyear']).mean()
#gradyear
#2006    18.655858
#2007    17.706172
#2008    16.767701
#2009    15.819573
#Name: age, dtype: float64

age_ye = snsData['age'].groupby(snsData['gradyear']).mean()
age_ye
#gradyear
#2006    18.655858
#2007    17.706172
#2008    16.767701
#2009    15.819573
#Name: age, dtype: float64

type(age_ye)#Series
age_ye.index
age_ye.values
age_ye[2006]
age_ye.loc[2006]
snsData['age'][2] is None

[age_ye[i] for i in snsData['gradyear']]

snsData['agemean'] = [age_ye[i] for i in snsData['gradyear']]

snsData.head(100)

#numpy에서 R의 ifelse와 비슷한 함수가 있다.
np.where(snsData['age'].isnull(),snsData['agemean'],snsData['age'])
#나이 평균값을 넣자 
snsData['age'] = np.where(snsData['age'].isnull(),
                   snsData['agemean'],snsData['age'])

#내가 했던 방식, 나이 평균값을 넣자 
for i in range(len(snsData)):
    if snsData['age'][i] is None:
        snsData['age'][i] = age_ye[snsData['gradyear'][i]]
  
#확인해보자, 여러 값들이 달라지는데 24477로 나누는걸
#30000으로 나누어서 그렇다.
snsData['age'].describe()
#count    30000.000000
#mean        17.237326
#std          1.141821
#min         13.027000
#25%         16.282000
#50%         17.238000
#75%         18.212000
#max         19.995000
#Name: age, dtype: float64

#5개 그룹으로 나누어보자 
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 5)
snsData.iloc[:,4:40].columns

##
from sklearn.preprocessing import scale
import numpy as np
train_data = np.array(scale(snsData.iloc[:,4:40]))
train_data

model.fit(train_data)
model.labels_
type(model.labels_)
Counter(model.labels_)
#Counter({3: 21323, 4: 5960, 0: 1038, 2: 872, 1: 807})
model.cluster_centers_

##표준화를 하자 
from sklearn.preprocessing import StandardScaler
data = snsData.iloc[:,4:40]
data_scale = StandardScaler().fit_transform(data)

model = KMeans(n_clusters = 5)
model.fit(data_scale)
model.labels_
model.cluster_centers_

snsData['cluster'] = model.labels_
snsData['cluster'].value_counts()
#2    22299
#0     5759
#1     1121
#3      820
#4        1
#Name: cluster, dtype: int64
#어 좀 이상한데, 한 그룹은 원소가 1개이다.
#0    21468
#3     6025
#4     1035
#1      872
#2      600
#Name: cluster, dtype: int64
#나중에 한 번 더 해보았을때 결과 

snsData.loc[0:5,['cluster','gender','age','friends']]

snsData['age'].groupby(snsData['cluster']).mean()
snsData['female'].groupby(snsData['cluster']).mean()
snsData['friends'].groupby(snsData['cluster']).mean()

#시각화를 해 보자 
import matplotlib.pyplot as plt
colormatp = np.array(['red','blue','green','yellow','black'])
plt.scatter(snsData.iloc[:,4],snsData.iloc[:,5],
            c = colormatp[model.labels_], s = 30)

centers = pd.DataFrame(model.cluster_centers_)
plt.scatter(centers.iloc[:,0], centers.iloc[:,1], s = 50,
            marker = 'D', c='g')
plt.show()

