# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:34:22 2019

@author: stu12
"""

#metacritic에서 점수를 긁어오자 
#게임이름, metascore, userscore, 출시년도를 저장하자
#그리고 그 정보를 csv file로 저장해두자 
#출시년도는 2016 ~ 2018년도로만 한정하자 
from bs4 import BeautifulSoup
import urllib.request as req

header = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64)\
          AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97\
          Safari/537.36",
          "Referer": "https://www.metacritic.com/browse/games/score/metascore/\
          year/all/filtered?sort=desc&year_selected={}&page={}"
          }

url = "https://www.metacritic.com/browse/games/score/metascore/year/all/\
filtered?sort=desc&year_selected={}&page={}"#출시년도,페이지 번호

search_years = range(2013,2020)

import copy

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

metascoreDF = DataFrame(columns = ['name','platform','metascore','userscore','year'])
metascoreDF

pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',10)

import re

#year는 숫자형으로 변환하는게 좋을까? 
pd.to_datetime(metascoreDF['year'],format = '%Y')


for search_year in search_years:
    print('year : {}'.format(search_year))
    page = 0
    while(True):
        print('page : {}'.format(page))
        temp_header = copy.deepcopy(header)
        temp_header['Referer'] = temp_header['Referer'].format(search_year,page)
        
        res = req.urlopen(req.Request(url.format(search_year,page),headers = temp_header))
        soup = BeautifulSoup(res,'html.parser')
        #body부분으로 범위를 줄인 다음에 
        soup_body = soup.find('div',class_='body_wrap')
        #page내용이 없는 경우 
        if (soup_body.find('p',class_='no_data') is not None):
            break
        start = len(metascoreDF)
        #게임 이름과 작동 기기 
        j = start
        for i in soup_body.find_all('div',class_='basic_stat product_title'):
            temp = i.get_text().strip()
            name = temp.split('(')[0].strip()
            platform = temp.split('(')[1].strip(')')
            metascoreDF.at[j,'name'] = name
            metascoreDF.at[j,'platform'] = platform
            j += 1
        #metascore
        j = start
        for i in soup_body.select('div.basic_stat.product_score.brief_metascore'):
            metascore = i.get_text().strip()
            metascoreDF.at[j,'metascore'] = metascore
            j +=1
        #userscore와 출시날짜(년도) 
        j = start
        for i in soup_body.find_all('li',class_= 'stat product_avguserscore'):
            temp = i.get_text().strip()
            userscore = temp.split(':')[1].strip()
            if userscore != 'tbd':
                metascoreDF.at[j,'userscore'] = userscore
            else:
                pass
            metascoreDF.at[j,'year'] = search_year
            j += 1
              
        page += 1

#userscore가 tbd인 row는 삭제할까?

metascoreDF.info()
metascoreDF.isnull().sum()
metascoreDF['year'].unique()

metascoreDF['metascore'] = metascoreDF['metascore'].astype(int)
metascoreDF['userscore'] = metascoreDF['userscore'].astype(float)


#userscore가 없는 (tbd) 이였던 부분은 각 metascore별로 평균을 구해서 넣자 
mean_userscore = metascoreDF['userscore'].groupby(metascoreDF['metascore']).mean()
mean_userscore.index
mean_userscore.values
metascoreDF['mean_userscore'] = [mean_userscore[i] for i in metascoreDF['metascore']]

metascoreDF['userscore'] = np.where(metascoreDF['userscore'].isnull(),
               metascoreDF['mean_userscore'],metascoreDF['userscore'])

#뽑아놓은거 저장하자 
metascoreDF.to_csv("C:/WorkSpace/PythonSpace/Python_Space/Project/metascore.csv")
metascoreDF.to_csv("C:/WorkSpace/PythonSpace/Python_Space/Project/metascore2019.csv")

import matplotlib.pylab as plt
#한글이 깨져서 나온다. 폰트에 한글폰트를 설정하자 
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

plt.scatter(metascoreDF.iloc[:,2],metascoreDF.iloc[:,3]*10)


#metascore로 k-mean 을 해 보자 
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 4)

sample = metascoreDF.iloc[:,2:4]
type(sample)
sample['userscore'] = sample['userscore']*10
model.fit(sample)
model.labels_
model.cluster_centers_

colormatp = np.array(['red','blue','green','yellow'])
plt.scatter(sample.iloc[:,0],sample.iloc[:,1],
            c = colormatp[model.labels_], s = 10)

#그외 ...

#########################################################################
#오픈크래딧에서 데이터를 추출하자 

header = {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/\
          537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36",
          "Referer":"https://opencritic.com/browse/all/{}?page={}"}
url = "https://opencritic.com/browse/all/{}?page={}"
    
opencriticDF = DataFrame(columns = ['openscore','name','platform','year'])
opencriticDF

for search_year in search_years:
    print('year : {}'.format(search_year))
    page = 1
    while(True):
        print('page : {}'.format(page))
        temp_header = copy.deepcopy(header)
        temp_header['Referer'] = temp_header['Referer'].format(search_year,page)
        res = req.urlopen(req.Request(url.format(search_year,page),headers = temp_header))
        soup = BeautifulSoup(res,'html.parser')
        
        soup_body = soup.find('div',class_='desktop-game-display')
        
        #내용이 없으면 멈춘다.
        if soup_body.get_text() == '':
            break
        
        start = len(opencriticDF)
        end = 0
        #점수
        j = start
        for i in soup_body.find_all('div',class_='score col-auto'):
            score = i.get_text().strip()
            #점수가 없으면 멈춘다.
            if score == '':
                end = j
                break
            opencriticDF.at[j,'openscore'] = score
            j += 1
        print('end : {}'.format(end)) 
        
        #게임이름
        j = start
        for i in soup_body.select('div.game-name.col > a'):
            name = i.get_text().strip()
            opencriticDF.at[j,'name'] = name
            j += 1
            if j == end:
                break
            
        #플랫폼
        j = start
        for i in soup_body.find_all('div',class_='platforms col-auto'):
            platform = i.get_text().strip()
            opencriticDF.at[j,'platform'] = platform
            #출시년도 
            opencriticDF.at[j,'year'] = search_year
            j += 1
            if j == end:
                break
        if end > 0:
            break
        page += 1
        
opencriticDF.info()
opencriticDF['openscore'] = opencriticDF['openscore'].astype('int')

#저장하자 
opencriticDF.to_csv("C:/WorkSpace/PythonSpace/Python_Space/Project/opencritic.csv")
opencriticDF.to_csv("C:/WorkSpace/PythonSpace/Python_Space/Project/opencritic2019.csv")

##metacritic에서 Genre 별로 
header = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64)\
          AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97\
          Safari/537.36",
          "Referer": "https://www.metacritic.com/browse/games/genre/metascore/\
          {}/all?view=condensed&page={}"
          }


url = "https://www.metacritic.com/browse/games/genre/metascore/\
{}/all?view=condensed&page={}"#genre, page 수

Genre = ['action','fighting','first-person','flight','party','puzzle','racing',
         'real-time','role-playing','simulation','sports','strategy',
         'third-person','turn-based','wargame','wrestling']

metaGenre = DataFrame(columns = ['name','platform','genre'])

page = 1

import copy
  
tempGenre = ['simulation','sports','strategy',
             'third-person','turn-based','wargame','wrestling']
   
for genre in tempGenre:
    print("장르 : {}".format(genre))
    page = 0
    while(True):
        print("page : {}".format(page))
        temp_header = copy.deepcopy(header)
        temp_header['Referer'] = temp_header['Referer'].format(genre,page)
        res = req.urlopen(req.Request(url.format(genre,page),headers = temp_header))
        soup = BeautifulSoup(res,'html.parser')

        soup_body = soup.find('div',class_='body')
        
        #page내용이 없는 경우 
        if (soup_body.find('p',class_='no_data') is not None):
            break
        
        start = len(metaGenre)
        j = start
        #name, platform, genre
        for i in soup_body.find_all('div',class_='basic_stat product_title'):
            temp = i.get_text().strip()
            name = temp.split('(')[0].strip()
            platform = temp.split('(')[1].strip(')')
            metaGenre.at[j,'name'] = name
            metaGenre.at[j,'platform'] = platform
            metaGenre.at[j,'genre'] = genre
            j += 1
        page += 1

metaGenre   
metaGenre['name']
metaGenre.info()
metaGenre.nunique()#이름이 겹치는 게 있다. 아마 여러 장르에 걸쳐 있는듯 하다. 
metaGenre['genre'].unique()

#장르는 더미 컬럼을 만들자
for j in range(len(metaGenre['genre'].unique())):
    temp = metaGenre['genre'].unique()[j]
    metaGenre[temp] = [1 if i == temp else 0 for i in metaGenre['genre']]
 
metaGenre.iloc[:,3:]

metaGenre[['name','platform']].nunique()

metaGenre.to_csv("c:/WorkSpace/Python_Space/Project/genre.csv")

#name중심으로 병합하자 
metaGenre2 = metaGenre.groupby(['name','platform'])[Genre].mean()
metaGenre2.index[1][0]

metaGenre2['platform'] = [metaGenre2.index[i][1] for i in range(len(metaGenre2))]
metaGenre2['name'] = [metaGenre2.index[i][0] for i in range(len(metaGenre2))]
metaGenre2.index = range(len(metaGenre2))

#장르중 값이 1 또는 0 이 아닌게 있는데...??? 아무래도 여러 장르에 중복으로 
#속해있던 게임인듯 하다. 0이 아닌것은 모두 1로 바꾸어 주자... 나중에 
#합치고 나서 .... 
metaGenre2.info()
metaGenre2['name'].sort_values()



metaGenre2.to_csv("c:/WorkSpace/Python_Space/Project/genre2.csv")

    
### 불러오기 ###
metascoreDF = pd.read_csv("c:/WorkSpace/Python_Space/Project/metascore.csv")
metascoreDF = metascoreDF.iloc[:,1:]

opencriticDF = pd.read_csv("c:/WorkSpace/Python_Space/Project/opencritic.csv")
opencriticDF = opencriticDF.iloc[:,1:]

metascore2019 = pd.read_csv("c:/WorkSpace/Python_Space/Project/metascore2019.csv")
metascore2019 = metascore2019.iloc[:,1:]

opencritic2019 = pd.read_csv("c:/WorkSpace/Python_Space/Project/opencritic2019.csv")
opencritic2019 = opencritic2019.iloc[:,1:]
### 

metascoreDF
metascore2019
#모두 합쳐서 merge까지 한 다음에 2018년도까지와 2019년을 나누자 

metascoreAll = metascoreDF.iloc[:,:-1].append(metascore2019.iloc[:,:-1])
metascoreAll.info()
metascoreAll.groupby('metascore')['userscore'].mean()

mean_userscore = metascoreAll.groupby('metascore')['userscore'].mean()
metascoreAll['mean_userscore'] = [mean_userscore[i] for i in metascoreAll['metascore']] 
metascoreAll['userscore'] = np.where(metascoreAll['userscore'].isnull(),
               metascoreAll['mean_userscore'],metascoreAll['userscore'])

#그리고 genre 도 합치자, genre에 없는 게임들이 있으니 left outer join 을 하자 
metascoreAll2 = pd.merge(metascoreAll, metaGenre2,
                         on = ['name','platform'] , how = 'left')

metascoreAll2.info()#null 값이 있네 다 0 을 넣자
metascoreAll2.fillna(0, inplace = True)

metascoreAll2.groupby(metascoreAll2['year']).count()
metascoreAll2.count()
metascoreAll2.nunique()
metascoreAll2[metascoreAll2['metascore'] >= 90]

opencriticAll = opencriticDF.append(opencritic2019)
opencriticAll.groupby(opencriticAll['year']).count()

#플랫폼마다 각기 따로 등록이 되어 있으니 이름을 합치자 
metascoreAll2.groupby('name')[['metascore','userscore']].mean()
metascoreDF_NAME = metascoreAll2.groupby('name')\
[['metascore','userscore','year','mean_userscore','action','fighting',
  'first-person','flight','party','puzzle','racing','real-time',
  'role-playing','simulation','sports','strategy','third-person',
  'turn-based','wargame','wrestling']].mean()
metascoreDF_NAME['name'] = metascoreDF_NAME.index
metascoreDF_NAME.index = range(len(metascoreDF_NAME))
metascoreDF_NAME.nunique()

#이제 장르가 1,0 이 아닌것들을 수정하자 
import collections
collections.Counter(metascoreDF_NAME['action'])
for i in range(len(Genre)):
    metascoreDF_NAME[Genre[i]] = [1 if j > 0 else 0 for j in metascoreDF_NAME[Genre[i]]]


metascoreDF_NAME.groupby('year').count()#출시년도가 정수로 안 떨어지는 건?
#어떻게 처리를 할까? 처음 출시년도를 기준으로 하자 
#goty는 최초 출시된 년도를 기준으로 수여하니까 
import math
metascoreDF_NAME['year'] = [math.floor(i) for i in metascoreDF_NAME['year']]

#metacritic을 기준으로 본 년도별 출시된 게임의 수 
year_count = collections.Counter(metascoreDF_NAME['year'])
type(year_count)
year_count.keys()
year_count.values()
plt.bar(year_count.keys(), year_count.values())

#년도별 점수 비교 
metascoreDF_NAME.groupby('year')[['metascore','userscore']].mean()
metascoreDF_NAME.groupby('year')[['metascore','userscore']].describe()

#저장
metascoreDF_NAME.to_csv("C:/WorkSpace/Python_Space/Project/metascoreDF_NAME.csv")
opencriticAll.to_csv("C:/WorkSpace/Python_Space/Project/opencriticAll.csv")
#############################################################
meta1 = metascoreDF_NAME[['name','metascore','userscore','year','action',
                          'fighting','first-person','flight','party','puzzle',
                          'racing','real-time','role-playing','simulation',
                          'sports','strategy','third-person','turn-based',
                          'wargame','wrestling']]
meta1.info()
open1 = opencriticAll[['openscore','name','year']]
open1
open1.columns
open1.info()


mergeDF = pd.merge(open1, meta1, on = 'name')
mergeDF.info()
mergeDF

(mergeDF['year_x'] == mergeDF['year_y']).sum()#년도가 다른게 있는데..... 

mergeDF.to_csv("C:/WorkSpace/Python_Space/Project/merge.csv")

