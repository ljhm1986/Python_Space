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
