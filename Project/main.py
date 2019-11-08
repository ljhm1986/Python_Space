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

search_years = [2016,2017,2018]

import copy
temp_header = copy.deepcopy(header)
temp_header
temp_header['Referer'] = temp_header['Referer'].format(2018,5)
res = req.urlopen(req.Request(url.format(2018,5),headers = temp_header))
soup = BeautifulSoup(res,'html.parser')
soup
type(soup)

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

metascoreDF = DataFrame(columns = ['name','platform','metascore','userscore','year'])
metascoreDF

pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',10)

#body부분으로 범위를 줄인 다음에 
soup_body = soup.find('div',class_='body_wrap')
soup_body

#게임 이름과 작동 기기 
j = 0
for i in soup_body.find_all('div',class_='basic_stat product_title'):
    temp = i.get_text().strip()
    name = temp.split('(')[0].strip()
    platform = temp.split('(')[1].strip(')')
    metascoreDF.at[j,'name'] = name
    metascoreDF.at[j,'platform'] = platform
    j += 1
    
metascoreDF['name']
metascoreDF['platform']


soup_body.find_all('div',class_= 'metascore_w small game positive')
soup_body.select('div.basic_stat.product_score.brief_metascore')
#metascore
j = 0
for i in soup_body.select('div.basic_stat.product_score.brief_metascore'):
    metascore = i.get_text().strip()
    metascoreDF.at[j,'metascore'] = metascore
    j +=1

sum(metascoreDF['metascore'].isnull())

#userscore
j = 0
for i in soup_body.find_all('li',class_= 'stat product_avguserscore'):
    temp = i.get_text().strip()
    userscore = temp.split(':')[1].strip()
    metascoreDF.at[j,'userscore'] = userscore
    j += 1
    
sum(metascoreDF['userscore'].isnull())    
    
metascoreDF[['name','userscore']]

import re
#출시날짜(년도) 
j = 0
for i in soup_body.select('li.stat.release_date.full_release_date > span.data'):
    #print(i.get_text().strip())
    #년도만 뽑자
    temp_text = i.get_text().strip()
    year = re.search('\d{4}',temp_text)
    metascoreDF['year'] = year[0]
    j += 1

metascoreDF.iloc[55,:]
metascoreDF.info()
metascoreDF['metascore'] = metascoreDF['metascore'].astype(int)
metascoreDF['userscore'] = metascoreDF['userscore'].astype(float)
len(metascoreDF)
(metascoreDF)
#year는 숫자형으로 변환하는게 좋을까? 
pd.to_datetime(metascoreDF['year'],format = '%Y')


for search_year in search_years:
    print('year : {}'.format(search_year))
    page = 0
    while(True):
        print('page : {}'.format(page))
        temp_header = copy.deepcopy(header)
        temp_header
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
        #userscore
        j = start
        for i in soup_body.find_all('li',class_= 'stat product_avguserscore'):
            temp = i.get_text().strip()
            userscore = temp.split(':')[1].strip()
            if userscore != 'tbd':
                metascoreDF.at[j,'userscore'] = userscore
            else:
                pass
            j += 1
        #출시날짜(년도) 
        j = start
        for i in soup_body.select('li.stat.release_date.full_release_date > span.data'):
            temp_text = i.get_text().strip()
            year = re.search('\d{4}',temp_text)
            metascoreDF['year'] = year[0]
            j += 1
        
        page += 1

#userscore가 tbd인 row는 삭제할까?

metascoreDF.info()
metascoreDF.isnull().sum()

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
metascoreDF.to_csv("C:\WorkSpace\Python_Space\Project\metascore.csv")



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

temp_header = copy.deepcopy(header)
temp_header
temp_header['Referer'] = temp_header['Referer'].format(2018,46)
res = req.urlopen(req.Request(url.format(2018,46),headers = temp_header))
soup = BeautifulSoup(res,'html.parser')

soup_body = soup.find('div',class_='desktop-game-display')

soup_body.get_text() == ''

#점수
for i in soup_body.find_all('div',class_='score col-auto'):
    print(i.get_text().strip())
    score = i.get_text().strip()
    if score == '':
        break
    
#점수 없는거 어떡?
    
#게임이름
for i in soup_body.select('div.game-name.col > a'):
    print(i.get_text().strip())

    
#플랫폼
for i in soup_body.find_all('div',class_='platforms col-auto'):
    print(i.get_text().strip())
    
#출시년도
#years 로
    
opencriticDF = DataFrame(columns = ['openscore','name','platform','year'])
opencriticDF

for search_year in search_years:
    print('year : {}'.format(search_year))
    page = 1
    while(True):
        print('page : {}'.format(page))
        temp_header = copy.deepcopy(header)
        temp_header
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
opencriticDF.to_csv("C:\WorkSpace\Python_Space\Project\opencritic.csv")

######
meta1 = metascoreDF[['name','metascore','userscore','year']]
meta1.info()
open1 = opencriticDF[['openscore','name','year']]
open1
open1.columns
open1.info()

meta2 = meta1.groupby(meta1['name']).mean().sort_values(by = 'metascore')
type(meta2)
meta2.columns
meta2.index
meta2['name'] = meta2.index
meta2.index = range(len(meta2))
meta2.info()


mergeDF = pd.merge(open1, meta2, on = 'name', how = 'outer')
mergeDF.info()

