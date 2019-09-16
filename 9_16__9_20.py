# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:23:37 2019

@author: stu11
"""

#9/16#
#데이터 분석을 할때 R에서 사용했던 자료형과 비슷한것들을 배워보자 
#pandas
#- 데이터분석 기능을 제공하는 라이브러리 
#- 1차원 배열 : Series(R : Vector)
#- 2차원 배열 : DataFrame(R : DataFrame)

from pandas import Series, DataFrame
#그외 pandas에서 제공하는 함수등을 사용하기 위해서 
import pandas as pd

#Series
#- 1차원 배열
#- 인덱스(색인) 배열의 데이터에 연관된 이름을 가지고 있다. 
#(pl/sql에서 index 배열을 기억하면...)
#- R의 백터 자료형하고 유사하다. 

s1 = Series([10,20,30,40,50])
s1
type(s1)
s1.astype#내부의 성분값들과 데이터 타입이 나온다.

s2 = Series(['10',20,30,40,50])
s2
type(s2)
s2.astype
#문자열을 object로 나타낸다. 

s2 + 10#문자타입이 있어서 error
s2 = s2.astype('int')#데이터타입을 int로 바꾸자 
s2 + 10

### index 전체에 연산하는 것은 list에서는 에러 ###
lst = [10,20,30]
lst + 10
###
[i + 10 for i in lst]
###

#index 정보 확인하기 
s2.index#RangeIndex(start=0, stop=5, step=1)
#값만 확인하기
s2.values
#index를 수정하기 
s2.index = ['a','b','c','d','e']
s2.index#Index(['a', 'b', 'c', 'd', 'e'], dtype='object')

s2 - 1
s2*2
s2 / 4
s2 % 4
s2 //4

#해당 index에 대해 value를 확인하기
s2['a']
s2[['a','c']]
s2[0]
s2[1:4]
s2[-1]
#20보다 큰 값들을 확인하자 
s2 > 20
s2[s2 > 20]

#해당값이 index안에 있는가
'a' in s2
'f' in s2

#해당 index의 value를 수정하기 
s2['a'] = 100
#해당 index와 그 값을 추가하기
s2['f'] = 500
#해당 index를 삭제하기 
del s2['f']
s2


s2['a'] = ''
s2.astype#값이 object로 바뀌어 있다. 주의하자 
s2 = s2.astype('int')


lst = [10,20,30]
#lst + 10
#list를 가지고 Series를 만든다. 
s3 = Series(lst)
s3

#tuple을 가지고 Series를 만든다.
tuple = (10,20,30)
s4 = Series(tuple)
s4

#dictionary를 가지고 Series를 만든다.
dict = {'a':1,'b':2,'c':3}
s5 = Series(dict)
s5

#특정 index만 골라서 만들어 보자 
ix = {'a','c'}
s6 = Series(dict, index = ix)
s6

ix = {'a','b','c','d'}
s7 = Series(dict, index = ix)
s7
#d의 value가 NaN이 낭ㄴ다.
#NaN : 인덱스 값을 찾을 수 없기 때문에 NaN으로 저장

import pandas as pd
pd.isnull(s7)
s7[pd.isnull(s7)]
s7[pd.notnull(s7)]

s8 = Series(range(10,60,10))
s8.index = ['a','b','c','d','e']
s8
#a    10
#b    20
#c    30
#d    40
#e    50
#dtype: int64

s8.astype
s8.dtype
s8['a'] = ''
s8.astype#object type으로 바뀐다.
s8.dtype

pd.isnull(s8)#이것으로는 index 모두 false가 농ㄴ다.
s8 == ''#하나 True로 나온다.

s9 = Series(range(10,60,10))
s9.index = ['a','b','c','d','e']
s9
s9['a']#nan

import numpy as np
s9['a'] = np.NaN
s9
s9.dtype
s9.astype
s9 + 10
pd.isnull(s9)
pd.notnull(s9)
s9 == ''

dict = {'서울': 500, '부산': 400,'경기':600,'제주':200}
s10 = Series(dict)
s10

city = {'서울','경기','제주','대전'}
s11 = Series(dict, index = city)
s11#대전 row는 NaN 들어있

s10 + s11#대전, 부산 : NaN
#(밑에 나오는)dtype옆에 붙이기
s11.name = '인구수'
s11
#index에 이름 추가 
s11.index.name = '지역명'
s11

#DataFrame : 2차원 배열
#- 각 컬럼의 서로 다른 종류 값을 표현 (문자, 숫자, 불리언)
#- data.frame
df1 = DataFrame([[1,2,3],[4,5,6],[7,8,9]])
df1

data = {'도시':['서울','부산','강원','인천'],
        '인구수':[500,400,200,300]}
data
type(data)
df2 = DataFrame(data)
df2

type(df2)
df2.astype
df2.dtypes
df2.columns
df2.index
df2.values

#해당 column의 값을 보는 방법은 ? (R : data.frame$컬럼이름)
df2.columns
df2.columns = ['지역','가구수']
df2.지역
df2.가구수
df2['지역']
df2['가구수']
df2['가구수'] * 10

#0번 row만 보자 
df2.ix[0]#loc 또는 iloc 를 사용하기를 권하는 메시지가 나온다.
df2.loc[0]
df2.iloc[0]

df2.index = ['a','b','c','d']
df2

#다음 둘의 결과는 같다.
df2.ix[0]
df2.ix['a']

df2.loc[0] #error, 인덱스 번호를 가지고 찾을 수 없다. 
df2.loc['a'] #인덱스 이름을 가지고 찾는다

df2.iloc[0] #인덱스 번호를 가지고 찾는다. 
df2.iloc['a'] # 인덱스 이름을 가지고 찾을 수 없다. 

s = Series([1000,200,3000,400],
           index = ['a','b','c','d'])
s
df2['부채'] = s
df2
del df2['부채']

data = {'서울':{2001:200,2002:300},
        '부산':{2000:100,2001:50,2002:200}}

data
df4 = DataFrame(data)
df4

df4.서울
df4.index
df4.values
#전
df5 = df4.T
df5
df5.dtypes

df4.dtypes
df4['서울'].dtype
df4.info
df4.info()

#index를 지정해서 Series를 만들때 row가 index순서대로 만들어 지지 않는다.
dict = {'a':10,'b':20,'c':30,'d':40}
ix = {'a','b','c','d'}
s = Series(dict, index = ix)
s
#reindex : 새로운 색인에 맞도록 객제를 새로 생성하는 기능,
#list와 DataFrame에서 사용된다.

#다음과 같이 하면 순서대로 만들어 진다.
obj = Series([10,20,30,40], index = ['c','d','a','b'])
obj

obj1 = obj.reindex(['a','b','c','d'])
obj1
obj2 = obj.reindex(['a','b','c','d','f'])
obj2#f는 NaN
obj3 = obj.reindex(['a','b','c'])
obj3#a,b,c 만 있음
obj4 = obj.reindex(['a','b','c','d','f'],fill_value = 0)
obj4#f는 0

list(range(4))

import numpy as np
np.arange(4)
#reshape(행,열)
np.arange(4).reshape(2,2)
np.arange(9).reshape(3,3)
np.arange(12).reshape(4,3)

df = DataFrame(np.arange(9).reshape(3,3),
               index=['a','b','c'], columns=['x','y','z'])
df
df2 = df.reindex(['a','b','c','d'])
df2#d row는 모두 NaN으로 값이 채워진다.
#method = 'ffill' or 'pad' : 앞줄을 기준으로 채우기
df3 = df.reindex(['a','b','c','d'], method = 'ffill')
df3#d row는 모두 c row의 값으로 채워진다.
df4 = df.reindex(['a','b','c','d'], method = 'pad')
df4#d row는 모두 c row의 값으로 채워진다.

##다시 보면 
df = DataFrame(np.arange(9).reshape(3,3),
               index=['a','b','c'], columns=['x','y','z'])
df
#   x  y  z
#a  0  1  2
#b  3  4  5
#c  6  7  8
df.reindex(['a','b','c','d'])
#     x    y    z
#a  0.0  1.0  2.0
#b  3.0  4.0  5.0
#c  6.0  7.0  8.0
#d  NaN  NaN  NaN
df.reindex(['a','b','c','d'],fill_value=0)
#   x  y  z
#a  0  1  2
#b  3  4  5
#c  6  7  8
#d  0  0  0 
#bfill, backfill : 뒷 줄을 기준으로 채우기
df.reindex(['a','b','c','d'], method = 'bfill')
df.reindex(['a','b','c','d'], method = 'backfill')
#     x    y    z
#a  0.0  1.0  2.0
#b  3.0  4.0  5.0
#c  6.0  7.0  8.0
#d  NaN  NaN  NaN

obj = Series(['sql','r','python'], index = [0,2,4])
obj
obj.reindex(range(6))
#0       sql
#1       NaN
#2         r
#3       NaN
#4    python
#5       NaN
#dtype: object
#fill_value : NaN자리에 넣을 값 
obj.reindex(range(6), fill_value = 'ML')
obj.reindex(range(6), fill_value = 'ffill')
#
obj.reindex(range(6), method = 'ffill')
obj.reindex(range(6), method = 'pad')
obj.reindex(range(6), method = 'bfill')
obj.reindex(range(6), method = 'backfill')

#Series 행 삭제하기 
obj = Series(np.arange(5),
             index = ['a','b','c','d','e'])
del obj['e']#바로 삭제
obj.drop('a')#삭제한걸 미리보기
obj.drop(['b','c'])

#DataFrame에서 행, 열 삭제하기
df = DataFrame(np.arange(16).reshape(4,4),
               index = ['w','x','y','z'],
               columns = ['one','two','three','four'])
df
#   one  two  three  four
#w    0    1      2     3
#x    4    5      6     7
#y    8    9     10    11
#z   12   13     14    15
df.drop('x',axis = 0) #axis = 0 행
df.drop('four',axis = 1) #axis = 1 열
df.drop(['y','z'], axis = 0)
df.drop(['one','two'],axis = 1)


obj = Series([10,20,30,40],index = ['a','b','c','d'])
obj
obj['a']
obj[0]
obj[1:3]
obj[['a','c']]


df['one']
df[['one','two']]
df[2]#error
df[2:]
df[0:2]
df[0] # error
df[0:1]

df.ix[0]
df.loc[0]#error
df.loc['w']
df.iloc['w']#error

df.ix['x','one']
df.loc['x','one']
df.iloc[1,'one']#error
df.ix[1,'one']
df.ix[1,['one','two']]
df.ix[['x','y'],['one','two']]
df.ix[[1,2]]
df.ix[0:2,0:2]
df.ix[:,0:2]
df.ix[-1,-1]

df < 5
#     one    two  three   four
#w   True   True   True   True
#x   True  False  False  False
#y  False  False  False  False
#z  False  False  False  False
df[df < 5]
#   one  two  three  four
#w  0.0  1.0    2.0   3.0
#x  4.0  NaN    NaN   NaN
#y  NaN  NaN    NaN   NaN
#z  NaN  NaN    NaN   NaN
df['one'] < 5
#w     True
#x     True
#y    False
#z    False
#Name: one, dtype: bool
df[df['one'] < 5]
#   one  two  three  four
#w    0    1      2     3
#x    4    5      6     7
df.ix[df['one'] < 5,0]
df.ix[df['one'] < 5,'one']
df.loc[df['one'] < 5,'one']


#[문제96] 아래와 같은 모양의 표를 생성하세요. 
#
#      PYTHON   R  SQL
#2014      60  90   50
#2015      80  65   75
#2016      70  75   85
from pandas import Series, DataFrame
#
df1 = DataFrame([[60,90,50],[80,65,75],[70,75,85]])
df1.columns = ['PYTHON','R','SQL']
df1.index = [2014,2015,2016]
df1
type(df1)
df1.astype
df1.dtypes
df1.columns
df1.index
df1.values
#
data1 = {'PYTHON':[60,80,70],'R':[90,65,75],'SQL':[50,75,85]}
data1
df2 = DataFrame(data1)
df2.index = [2014,2015,2016]
df2

#선생님 풀
df3 = DataFrame({'PYTHON':[60,80,70],'R':[90,65,75],'SQL':[50,75,85]},
                 index = ['2014','2015','2016'])
df3
#[문제97] 'PYTHON' 열을 선택하세요
#loc는 이름으로만 찾는다, iloc는 번호로만 찾는다.
df1.PYTHON
df1['PYTHON']
df1.ix[:,0]
df1.ix[:,'PYTHON']
df1.loc[:,'PYTHON']
df1.loc[:,0]#error
df1.iloc[:,0]
df1.iloc[:,'PYTHON']#error

#[문제98] '2014' 행 정보를 출력하세요.
#loc는 이름으로만 찾는다, iloc는 번호로만 찾는다.
df1.loc[2014]
df1.ix[2014]
df3.ix['2014']
df3.loc['2014']
df3.iloc[0]
df1.iloc[0]
df1.iloc[2014]#index에 숫자형을 넣으면 안 된다.
df1.ix[0]#index에 숫자형을 넣으면 혼동이 된다.
#index의 값이 숫자형으로 만들면 loc나 iloc로 찾는다.

#[문제99] 인덱스번호를 기준으로 1부터 2번까지 출력하세요.
df1.iloc[1:3]
df3.ix[1:3]
df3.iloc[1:3]

#[문제100] PYTHON의 값을 기준으로 60보다 큰값을 가지고 있는 행 정보를 출력하세요.
df1[df1['PYTHON'] > 60]

#[문제101] PYTHON의 값을 기준으로 60 보다 큰값을 가지고 있는 PYTHON 정보만 출력하세요.
df1[df1['PYTHON'] > 60]['PYTHON']
df3.ix[df3['PYTHON'] > 60, 'PYTHON']
df3.ix[df3['PYTHON'] > 60, 0]
df3.loc[df3['PYTHON'] > 60, 'PYTHON']
df3['PYTHON'][df3['PYTHON'] > 60]

#[문제102] '2015' 행값 중에 PYTHON 정보만 출력하세요
df1.ix[2015,'PYTHON']
df3.ix['2015','PYTHON']
df3.loc['2015','PYTHON']
df3.iloc[1,0]

#[문제103] '2015' 행값 중에 PYTHON, R 정보 출력하세요 
df1.ix[2015][['PYTHON','R']]
df3.ix['2015'][['PYTHON','R']]
df3.loc['2015'][['PYTHON','R']]
df3.iloc[1][[0,1]]

#[문제104] 'R' 열 정보를 출력하세요.
df1['R']
df1.ix[:,'R']
df1.loc[:,'R']
df1.iloc[:,1]

#at :
#DataFrame에서 새로운 행을 추가하기, 기존행이 있으면 수정
df3.at['2013','PYTHON'] = 100
df3
df3.at['2013','R'] = 90
df3.at['2013','SQL'] = 85
df3
#삭제하려면
df3 = df3.drop('2013',axis = 0)
df3

df3.set_value('2013','PYTHON',100)
df3.set_value('2013','R',90)
df3.set_value('2013','SQL',85)
df3

#iat는 수정만 가능하다. 값을 추가할 수 없다.
df3.iat[0,0]
df3.iat[0,1]
df3.iat[3,0] = 100#만약 row 없으면 안 
df3

#새로운 열을 추가하는 방법
df3.at['2013','JAVA'] = 50
df3
df3.at['2014','JAVA'] = 60
df3.at['2015','JAVA'] = 70
df3.at['2016','JAVA'] = 80
df3

#row의 순서를 수정하자 
df3 = df3.reindex(['2013','2014','2015','2016'])
df3

#column의 순서를 수정할 수도 있다
df3.reindex(columns = ['SQL','R','PYTHON','JAVA'])

#[문제 105] PYTHON점수가 80점 이상 또는 SQL점수가
#90점 이상인 데이터를 출력해주세요
(df3['PYTHON'] >= 80) | (df3['SQL'] >= 90)#각기 ()하고 | 해야 한다.
df3[(df3['PYTHON'] >= 80) | (df3['SQL'] >= 90)]

#[문제 106] PYTHON점수가 80점 이상이고 SQL점수가
#90점 이상인 데이터를 출력해주세요
df3[(df3['PYTHON'] >= 80) & (df3['SQL'] >= 90)]

