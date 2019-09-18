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
#0    10
#1    20
#2    30
#3    40
#4    50
#dtype: int64
type(s1)#pandas.core.series.Series
s1.astype#내부의 성분값들과 데이터 타입이 나온다.

s2 = Series(['10',20,30,40,50])
s2
#0    10
#1    20
#2    30
#3    40
#4    50
#dtype: object
type(s2)#pandas.core.series.Series
s2.astype
#문자열을 object로 나타낸다. 

s2 + 10#문자타입이 있어서 error
s2 = s2.astype('int')#데이터타입을 int로 바꾸자 
s2 + 10

### index 전체에 연산하는 것은 list에서는 에러 ###
lst = [10,20,30]
lst + 10
###다음과 같이 반복문을 사용해야한다.
[i + 10 for i in lst]
###

#index 정보 확인하기 
s2.index#RangeIndex(start=0, stop=5, step=1)
#값만 확인하기
s2.values#array([10, 20, 30, 40, 50])
#index를 수정하기 
s2.index = ['a','b','c','d','e']
s2.index#Index(['a', 'b', 'c', 'd', 'e'], dtype='object')

#다음 연산들이 Series 전체의 각 요소에 작용한다.
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

#해당값이 index안에 있는지 확인하기
'a' in s2
'f' in s2

#해당 index의 value를 수정하기 
s2['a'] = 100
#해당 index와 그 값을 추가하기
s2['f'] = 500
#해당 index를 삭제하기 
del s2['f']
s2

#
s2['a'] = ''
s2.astype#dtype이 int64에서 object로 바뀌어 있다. 주의하자 
s2 = s2.astype('int')
#ValueError: invalid literal for int() with base 10: ''


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
#d    NaN
#b    2.0
#c    3.0
#a    1.0
#dtype: float64
#d의 value가 NaN이 나온다.
#NaN : 인덱스 값을 찾을 수 없기 때문에 NaN으로 저장

import pandas as pd

#NaN인것 찾기
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

pd.isnull(s8)#이것으로는 index 모두 false가 나온다
s8 == ''#a row가 True로 나온다.

s9 = Series(range(10,60,10))
s9.index = ['a','b','c','d','e']
s9
s9['a']

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
s11#대전 row는 NaN 들어있다

s10 + s11#대전, 부산 : NaN
#(밑에 나오는)dtype옆에 붙이기
s11.name = '인구수'
s11
#index에 이름 추가 
s11.index.name = '지역명'
s11

##############################
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

type(df2)#pandas.core.frame.DataFrame

df2.astype
#DataFrame의 column의 데이터 타입을 보여준다.
df2.dtypes
#DataFrame의 column의 이름을 보여준다.
df2.columns
#DataFrame의 index의 값을 보여준다.
df2.index
#DataFrame의 성분을 보여준다.
df2.values

#해당 column의 값을 보는 방법은 ? (R : data.frame$컬럼이름)
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
#numpy 배열 생성 : 
#arange : 숫자 생성 , arange(숫자갯수) ,arange(처음숫자, 종료숫자, 간격)
np.arange(4)
#reshape(행,열) :행*열 형태를 갖춘다
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

##########################################################################
#9/17#
from pandas import Series, DataFrame
#[문제 107] student 데이터 프레임을 생성하세요.
student = DataFrame({'영어':[60,50,90],'수학':[80,70,80],'국어':[70,85,95]},
                    index = ['홍길동','박찬호','손흥민'])
student
#     영어  수학  국어
#홍길동  60  80  70
#박찬호  50  70  85
#손흥민  90  80  95
student.info()

student2 = DataFrame([[60,50,90],[80,70,80],[70,85,95]],
                     index = ['홍길동','박찬호','손흥민'],
                     columns = ['영어','수학','국어'])
student2

#[문제 108] student 데이터 프레임에 새로운 데이터를 생성하세요
student.at['제임스','영어'] = 100
student.at['제임스','수학'] = 50
student.at['제임스','국어'] = 80
student.info
#<bound method DataFrame.info of         영어    수학    국어
#홍길동   60.0  80.0  70.0
#박찬호   50.0  70.0  85.0
#손흥민   90.0  80.0  95.0
#제임스  100.0  50.0  80.0>
student.info()
#<class 'pandas.core.frame.DataFrame'>
#Index: 4 entries, 홍길동 to 제임스
#Data columns (total 3 columns):
#영어    4 non-null float64
#수학    4 non-null float64
#국어    4 non-null float64
#dtypes: float64(3)
#memory usage: 288.0+ bytes
#모두 float64 type으로 변해있다. 이유는 처음에 NaN이 생기는데
#다음에 무슨값이 들어올지 모르니까 int64를 float64로 바꾸어 놓는다.
#int로 하고 싶으면 다음과 같이 변형하자 
student.astype('int')

#[문제 109] 제임스의 영어 점수를 출력해주세요
student.ix['제임스']['영어']
student.loc['제임스']['영어']
student.iloc[3][0]
student.at['제임스','영어']
student.iat[3,0]

#2개의 DataFrame을 붙여보자, columns 이 같은 DataFrame을 만들어서  
student3 = DataFrame([[60,50,90],[80,70,80],[70,85,95]],
                     index = ['유재석','게롤드','도바킨'],
                     columns = ['영어','수학','국어'])
student3
student.append(student3)
student = student.append(student3)
#또는 pandas에서 제공하는 concat()을 사용하자
import pandas as pd
pd.concat([student, student3])

#3개이상을 합치려면 append는 2개씩 합치는걸 반복해야 한다.
#반면에 pd.concat은 list에 나열하면 된다.

student['일본어'] = [100,90,80,70]
student['한국사'] = 100#모든 row에 100으로 채워진다.
student['음악'] = np.nan#모든 row에 NaN을 넣는다.

#student에서 홍길동의 음악점수를 보기 
student.ix['홍길동']['음악']
student.ix[0][5]
student.loc['홍길동']['음악']
student.iloc[0][5]
student.at['홍길동','음악']
student.iat[0,5]

#student에서 박찬호의 점수들을 보기
student.ix['박찬호']
student.loc['박찬호']
student.at['박찬호']#error
student.xs('박찬호')
student.xs('박찬호', axis = 0)#row
#student에서 영어 점수들을 보
student.xs('영어',axis = 1)#column
student.ix[:,'영어']
student.loc[:,'영어']

#index이름 수정
student = student.rename(index={'박찬호':'찬호박'})
student

#column 이름 수정
student = student.rename(columns = {'수학':'산수'})
student

#csv file을 pandas로 불러오기
emp = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\emp.csv')
emp
#column의 type을 보자 
emp.dtypes
#
emp.info()

emp.ix[emp['EMPLOYEE_ID'] == 100,'SALARY']
emp.loc[emp['EMPLOYEE_ID'] == 100,'SALARY']
emp['SALARY'][emp['EMPLOYEE_ID'] == 100]

#[문제 110] JOB_ID가 ST_CLERK인 사원의 LAST_NAME, SALARY를 출력해주세요
emp.ix[emp['JOB_ID'] == 'ST_CLERK',['LAST_NAME','SALARY']]
emp.loc[emp['JOB_ID'] == 'ST_CLERK',['LAST_NAME','SALARY']]
emp[['LAST_NAME','SALARY']][emp['JOB_ID'] == 'ST_CLERK']

#[문제 111] SALARY가 10000 이상인 사원들의 LAST_NAME, SALARY를 출력해주세요
emp[emp.SALARY >= 10000][['LAST_NAME','SALARY']]

emp.ix[emp.SALARY >= 10000,['LAST_NAME','SALARY']]
emp.loc[emp.SALARY >= 10000,['LAST_NAME','SALARY']]


#Series 간의 연산
obj1 = Series([1,2,3,4,5], index = ['a','b','c','d','e'])
obj2 = Series([2,4,6,8,10],index = ['a','b','c','d','e'])

obj1 + obj2
obj1.add(obj2, fill_value = 0)

obj1 - obj2
obj1.sub(obj2, fill_value = 0)

obj1 * obj2
obj1.mul(obj2, fill_value = 0)

obj1 / obj2
obj1.div(obj2, fill_value = 0)

obj1 // obj2
obj2 // obj1
obj2 % obj1
obj2.mod(obj1, fill_value = 0)

obj1 ** obj2
obj1.pow(obj2, fill_value = 0)
##
import numpy as np
df1 = DataFrame(np.arange(6).reshape(2,3),
                index=['2015','2016'],
                columns=['python','sql','plsql'])
#      python  sql  plsql
#2015       0    1      2
#2016       3    4      5

df2 = DataFrame(np.arange(12).reshape(3,4),
                index=['2014','2015','2016'],
                columns=['python','r','sql','plsql'])
#      python  r  sql  plsql
#2014       0  1    2      3
#2015       4  5    6      7
#2016       8  9   10     11

df1 + df2
df1.add(df2, fill_value=0)
df1.sub(df2, fill_value=0)
df1.mul(df2, fill_value=1)
df1.div(df2, fill_value=1)
df2.mod(df1, fill_value=1)

#sort() : Series와 DataFrame을 정렬
obj = Series([2,3,7,8], index = ['d','a','b','c'])
obj
obj.reindex(['a','b','c','d'])
obj.sort_index() # index를 기준으로 오름차순 정렬 
obj.sort_index(ascending = False) # index를 기준으로 내차순 정렬 
obj.sort_values() # value를 기준으로 오름차순 정렬
obj.sort_values(ascending = False)# value를 기준으로 내림차순 정렬

df = DataFrame(np.arange(8).reshape(2,4),
               index = ['two','one'],
               columns = ['d','a','c','b'])
df
#     d  a  c  b
#two  0  1  2  3
#one  4  5  6  7
df.sort_index()
df.sort_index(ascending = False)
df.sort_index(axis = 0) # index, row
#     d  a  c  b
#one  4  5  6  7
#two  0  1  2  3
df.sort_index(axis = 1) # column
#     a  b  c  d
#two  1  3  2  0
#one  5  7  6  4
df.sort_index(ascending = False, axis = 1)
df.sort_values() # error
df.sort_values(by='b', axis=0, ascending=False)# b column 기준
#     d  a  c  b
#one  4  5  6  7
#two  0  1  2  3
df.sort_values(by='one', axis=1, ascending=False)# one row 기준
#     b  c  a  d
#two  3  2  1  0
#one  7  6  5  4

#[문제 112] SALARY가 10000이상인 사원들의 LAST_NAME, SALARY,
#DEPARTMENT_ID를 출력하세요. 단 DEPARTMENT_ID를 기준으로 
#오름차순 정렬하세요.

emp
type(emp)
emp1 = emp[emp.SALARY >= 10000][['LAST_NAME', 'SALARY','DEPARTMENT_ID']]
emp1.sort_index(by = 'DEPARTMENT_ID')
emp1.sort_values(by = 'DEPARTMENT_ID')

emp1.sort_values(by = ['DEPARTMENT_ID','SALARY'], ascending = [True, False])

##
obj = Series([78,80,88,60,50,90,79,99,68,80])
obj.sort_values()
obj.sort_values(ascending = False)

#오름차순으로 등수를 보자 
obj.rank()
#내림차순으로 등수를 보자 

obj.rank(ascending=False, method = 'average')
obj.rank(ascending=False, method = 'min')
obj.rank(ascending=False, method = 'max')
#등수가 같을때, 앞쪽에 있는걸 높게 함 
obj.rank(ascending=False, method = 'first')
#등수가 중간에 빈 숫자가 없게 함
obj.rank(ascending=False, method = 'dense')

#순위와 점수가 있는 DataFrame 만들기
obj = Series([78,80,88,60,50,90,79,99,68,80])
df = DataFrame({'순위':obj.rank(ascending=False, method = 'dense'),
                '점수':obj})
df.sort_values(by='순위')
df.astype('int')

obj = Series([78,80,88,np.nan, 90])
obj.sort_values()
obj.sort_values(ascending = False)
obj.sort_values(ascending = False, na_position = 'filst')

obj.rank()

obj.rank(appending=False)
obj.rank(na_option='keep')

df = DataFrame({'영어':[60,80,70],'수학':[50,60,70]},
                index=['홍길동','김건모','이문세'])

df.sort_values(by = '수학')
df['수학'].sort_values(ascending = False)
df.rank(ascending = False)
df.rank(ascending = False, axis = 1)
df['영어'].rank(ascending = False)
df.loc['홍길동'].rank(ascending=False)

#[문제 113] 급여를 많이 받는 순으로 10위까지 출력해주세요
#emp[emp['SALARY'].rank(ascending = False) <= 10]

emp1 = emp
emp1['RANK'] = emp1['SALARY'].rank(ascending=False, method='dense')
emp1[['RANK','EMPLOYEE_ID','SALARY']][emp['RANK'] <= 10].sort_values(by = 'RANK')

emp1['RANK2'] = emp1['SALARY'].rank(ascending=False, method='first')
emp1[['RANK2','EMPLOYEE_ID','SALARY']][emp['RANK2'] <= 10].sort_values(by = 'RANK2')

##수학 점수가 50, 60점인 사람
df['수학'].isin([50,60])
df[df['수학'].isin([50,60])]
#수학 점수가 50, 60점이 아닌 사람
df[~df['수학'].isin([50,60])]

#[문제 114] JOB_ID가 AD_VP, AD_PRES인 사원들의 LAST_NAME, SALARY, JOB_ID를 출력하세요
emp['JOB_ID'].isin(['AD_VP','AD_PRES'])
emp[emp['JOB_ID'].isin(['AD_VP','AD_PRES'])][['LAST_NAME','SALARY','JOB_ID']]
emp.loc[emp['JOB_ID'].isin(['AD_VP','AD_PRES']),['LAST_NAME','SALARY','JOB_ID']]

#[문제 114] JOB_ID가 AD_VP, AD_PRES가 아닌 사원들의 LAST_NAME, SALARY, JOB_ID를 출력하세요
emp[~emp['JOB_ID'].isin(['AD_VP','AD_PRES'])][['LAST_NAME','SALARY','JOB_ID']]
emp.loc[~emp['JOB_ID'].isin(['AD_VP','AD_PRES']),['LAST_NAME','SALARY','JOB_ID']]

#null 처리
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from numpy import nan as NA

obj1 = Series([1,2,3,None,5])
obj2 = Series([1,2,3,np.nan,5])
obj3 = Series([1,2,3,NA,5])
#null인 부분 
obj1.isnull()
obj2.isnull()
obj3.isnull()
#null이 아닌 부분
obj1.notnull()
obj2.notnull()
obj3.notnull()

obj1[obj1.isnull()]
obj1[obj1.notnull()]

pd.isnull(obj1)
pd.notnull(obj1)

obj1 = obj1.fillna(0)#nan를 0으로 채우기
obj2.dropna()#nan를 지우기, na row가 삭제된다.

df = DataFrame([[1,2,3],[1,None,NA],[NA,NA,NA]])
df
df.dropna()
#row가 전부 null인 row만 지우기 
df.dropna(how='all',axis=0)
df[2] = NA
df
#column이 전부 null인 column만 지우기 
df.dropna(how='all',axis=1)
#null에 전부 0 채우기
df.fillna(0)
df[0].fillna(0)
#column마다 채워야 할 값들을 다르게 해보자 
df.fillna({0:0,1:10,2:20})

#미리보기가 아니라 바로 값을 채워넣으려면 
df.fillna(0, inplace = True)
df

#df = DataFrame([[1,2,3],[1,None,NA],[NA,NA,NA]])
#앞의 값으로 채우기
df.fillna(method = 'ffill')
df.fillna(method = 'pad')
#뒤의 값으로 채우기 
df.fillna(method = 'bfill')
df.fillna(method = 'backfill')

#[문제 116] commission_pct가 null인 사원의 LAST_NAME, SALARY를 출력하세요
emp['COMMISSION_PCT'].isnull()
emp.loc[emp['COMMISSION_PCT'].isnull(),['LAST_NAME','SALARY']]
#[문제 117] commission_pct가 null이 아닌 사원의 LAST_NAME, SALARY를 출력하세요
emp.loc[emp['COMMISSION_PCT'].notnull(),['LAST_NAME','SALARY']]

#[문제 118] LAST_NAME 첫글자가 S로 시작되는 LAST_NAME을 출력하세요

[emp['LAST_NAME'][i] for i in range(len(emp)) if emp['LAST_NAME'][i][0].upper() == 'S']

for i in range(len(emp)):
    if emp['LAST_NAME'][i][0].upper() == 'S':
        print(emp['LAST_NAME'][i])
        
for i in emp['LAST_NAME']:
    if i[0] == 'S':
        print(i)

[i[0] == 'S' for i in emp['LAST_NAME']]
emp['LAST_NAME'][[i[0] == 'S' for i in emp['LAST_NAME']]]

emp.loc[[i[0] == 'S' for i in emp['LAST_NAME']],'LAST_NAME']

#R에서 apply계열 사용한것 처럼
s = Series([1,2,3])
s
s ** 2

def square(arg):
    return arg ** 2

square(s)
square(5)

#for문을 사용하지 않고 해보기 
#apply() : 행과 열값을 인수값으로 받아서 반복하며 함수를 적용한다.
#pandas에서 제공한다.
s.apply(square)
s.apply(lambda arg : arg ** 2)

df = DataFrame([[1,2,3],[4,5,6]])
df.apply(square)
df.apply(lambda arg : arg ** 2)

emp[emp['LAST_NAME'].apply(lambda arg : arg[0] == 'S')]['LAST_NAME']
emp.loc[emp['LAST_NAME'].apply(lambda arg : arg[0] == 'S'),'LAST_NAME']
#startswith()를 이용하면 
emp[emp['LAST_NAME'].apply(lambda x : x.startswith('S'))]['LAST_NAME']

#
def first_character_S(x):
    if x[0] == 'S':
        return True
    else:
        return False
    
emp['LAST_NAME'][[first_character_S(i) for i in emp['LAST_NAME']]]
emp['LAST_NAME'][emp['LAST_NAME'].apply(first_character_S)]

##########################################################################
#9/18#
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from numpy import nan as NA
#[문제 119] g로 끝나는 LAST_NAME을 출력하세요
emp = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\emp.csv')

emp['LAST_NAME'].apply(lambda x : x[-1] == 'g')
emp[emp['LAST_NAME'].apply(lambda x : x[-1] == 'g')]['LAST_NAME']
emp.loc[emp['LAST_NAME'].apply(lambda x : x[-1] == 'g')]['LAST_NAME']
#
emp[emp['LAST_NAME'].apply(lambda x : x.endswith('g'))]['LAST_NAME']
emp.loc[emp['LAST_NAME'].apply(lambda x : x.endswith('g'))]['LAST_NAME']
#
for i in emp['LAST_NAME']:
    if i[-1] == 'g':
        print(i)
#
[emp['LAST_NAME'][i] for i in range(len(emp)) if emp['LAST_NAME'][i][-1] == 'g']
[i for i in emp['LAST_NAME'] if i[-1] == 'g']
#
x = [i[-1] == 'g' for i in emp['LAST_NAME']]
emp['LAST_NAME'][x]
emp.loc[x,'LAST_NAME']

#함수에 입력변수가 2개일때 apply() 사용  
def last_character(x,y):
    if x[-1] == y:
        return True
    else:
        return False
[i for i in emp['LAST_NAME'] if last_character(i,'g')]
#
emp['LAST_NAME'][emp['LAST_NAME'].apply(last_character, y='g')]       
#

#[문제 120] 관리자 사원들의 LAST_NAME, SALARY를 출력하세요
emp.columns
emp['MANAGER_ID']
emp['EMPLOYEE_ID']
#1.pandas를 이용해서 해결
emp['EMPLOYEE_ID'].isin(emp['MANAGER_ID'])
emp[emp['EMPLOYEE_ID'].isin(emp['MANAGER_ID'])][['LAST_NAME','SALARY']]
#2.pandas를 사용하지 않고 해결
#emp 자체가 pandas가 필요했네...
for i in range(len(emp)):
    for j in range(len(emp)):
        if emp['EMPLOYEE_ID'][i] == emp['MANAGER_ID'][j]:
            print(emp['LAST_NAME'][i],emp['SALARY'][i])
            break
#선생님의 풀이 
import csv
#1. list를 하나 만들어서 manager_id를 저장하자 
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)
v_mgr = []
for i in emp_csv:
    if (not i[9] in v_mgr) & (i[9] != ''):
        v_mgr.append(i[9])
file.close()

#2. list에 포함되는 employee_id인 last_name, salary를 출력하자 
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)
for i in emp_csv:
    for mgr in v_mgr:
        if i[0]  == mgr:
            print(i[2],i[7])
            
file.close()
#

##
import numpy as np
s = Series([3,4,2,None,6])

#sum() : 합, nan 가 있으면 제외하고 계산을 한다.
s.sum()
s.sum(skipna = True) #default
s.sum(skipna = False) #
#평균 
s.mean()
#분산
s.var()
#표준편차
np.sqrt(s.var())
s.std()
#
s.max()
#
s.min()
#최대값
s.idxmax()
#최소값
s.idxmin()

s[s.idxmax()]
s[s.idxmin()]

#예전에 쓰던 것, 요즘 안 씀 
#s.argmax()
#s.argmin()
#누적 합
s.cumsum()
#누적 곱
s.cumprod()

#
s = Series([3,4,2,5,np.nan,6,2,6])
#해당 index까지 중에서 최소값
s.cummin()
#해당 index까지 중에서 최대값
s.cummax()

s.max()
s.min()

#보면 해당 index가 하나만 나온다.
s.idxmax()
s.idxmin()
s[s.idxmax()]
s[s.idxmin()]
#다음과 같이 하면 중복되는 최대,최소값들을 다 볼 수 있다.
s[s == s.max()]
s[s == s.min()]

s[s == s.max()].index
s[s == s.max()].values

s.count()#nan 제외한 건수
len(s)#nan 포함한 건수

#다음의 여러가지 값을 한 번에 출력함 
s.describe()
#count    7.000000
#mean     4.000000
#std      1.732051
#min      2.000000
#25%      2.500000
#50%      4.000000
#75%      5.500000
#max      6.000000
#dtype: float64

df = DataFrame([[44,65,93],[64,12,70],[98,43,33]],
               index = ['머신러닝','도바킨','게롤드'],
               columns = ['영어','수학','국어'])
df
#연산의 기본값이 각 column에 row를 더한다.
df.sum()
df.sum(axis = 0)
df.sum(axis = 'rows')

df.sum(axis = 1)
df.sum(axis = 'columns')

df.mean()

df.at['카카로트','영어'] = 84
df.at['카카로트','수학'] = np.nan
df.at['카카로트','국어'] = 90

df.sum()
df.mean()
df.mean(axis = 1, skipna = False)
df['영어'].sum()
df.describe()

s = Series([1,2,3,5,23,5,6,7,1,4,2,None,5,4,2,7,5,4,2,4,5,7,69,None,1,3,4,9,41,
            5,6,54,98,5,6])
s.unique()
s.value_counts()
s.value_counts(dropna = False)
s.value_counts(sort = True)
s.value_counts(sort = False)

pd.value_counts(s)
pd.value_counts(s,sort=False)

df = DataFrame({'a':['a1','a2','a3','a5','a2','a1'],
                'b':['b4','b2','b6','b4','b8',np.nan]})
df
df['a'].unique()
df['b'].unique()
df['a'].value_counts()
df['b'].value_counts()
df['b'].value_counts(dropna=False)

#[문제 121] 최고급여, 최저급여를 출력하세요
#1. pandas를 이용하는 방법
emp['SALARY'].max()
emp['SALARY'].min()
#2. pandas를 이용하지 않는 방법 
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)

max_value = emp['SALARY'][0]
for i in emp['SALARY']:
    if i > max_value:
        max_value = i
print(max_value)
file.close()

file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)

min_value = emp['SALARY'][0]
for i in emp['SALARY']:
    if i < min_value:
        min_value = i
print(min_value)
file.close()

#위에는 emp['SALARY'] 가 pandas 사용한 것...;; 다시 해보자 
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)

max_value = int(next(emp_csv)[7])
for i in emp_csv:
    if int(i[7]) > max_value:
        max_value = int(i[7])
print(max_value)
file.close()

file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)

min_value = int(next(emp_csv)[7])
for i in emp_csv:
    if int(i[7]) < min_value:
        min_value = int(i[7])
print(min_value)
file.close()

#옆자리 서민형씨 풀이
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)

lst = []
for i in emp_csv:
    lst.append(int(i[7]))
lst.sort(reverse = True)
print(lst[0])
print(lst[-1])
file.close()

#[문제 122] 20번 부서 사원들의 급여의 합을 구하세요
#1. pandas 이용해서 해결
emp.info()
emp['DEPARTMENT_ID'] == 20
emp[emp['DEPARTMENT_ID'] == 20]['SALARY'].sum()

#2. pandas 이용하지 않고 해결
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)

sum_salary = 0
for i in emp_csv:
    if i[-1] == '20':
        sum_salary += int(i[7])
print(sum_salary)
file.close()

#[문제 123] 부서번호를 입력하면 그 부서의 급여의 총액을 구하는 
#함수를 생성하세요
#dept_sum_sal(50)
#답..
#dept_sum_sal(1000)
#부서가 없습니다.
emp.info()
###
for i in emp:
    print(i)
#EMPLOYEE_ID
#FIRST_NAME
#LAST_NAME
#EMAIL
#PHONE_NUMBER
#HIRE_DATE
#JOB_ID
#SALARY
#COMMISSION_PCT
#MANAGER_ID
#DEPARTMENT_ID


def dept_sum_sal(dept_id):
    
    emp = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\emp.csv')
    lst = emp['DEPARTMENT_ID'].unique()
    #null인거 없에자 
    lst = lst[pd.notnull(lst)]
    
    if not dept_id in lst:
        print('부서가 없습니다.')
        return
    
    sum_salary = 0
    for i in range(len(emp)):
        if emp['DEPARTMENT_ID'][i] == dept_id:
            sum_salary += emp['SALARY'][i]
    print(sum_salary)
    
dept_sum_sal(20)
dept_sum_sal(40)
dept_sum_sal(200)
dept_sum_sal(None)
dept_sum_sal(np.nan)
dept_sum_sal(NA)

#선생님의 풀이 
def dept_sum_sal2(dept_id):
    
    emp = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\emp.csv')
    lst = emp['DEPARTMENT_ID'].unique()
    #null인거 없에자 
    lst = lst[pd.notnull(lst)]
    
    if dept_id in lst:
        return emp['SALARY'][emp['DEPARTMENT_ID'] == float(dept_id)].sum()
    else:
        print('부서가 없습니다.')
        return
    
dept_sum_sal2(20)
dept_sum_sal2(40)
dept_sum_sal2(200)

#[문제 124] s변수에 값들 중에 unique한 값만 s_unique 변수에 넣어주세요
s = [1,2,3,4,1,2,3,4,5,1,2,3,4,5,6,'']
s
type(s)
#set
s_unique = [i for i in set(s) if i != '']
s_unique
#list
s
s_unique = []
for i in s:
    if (not i in s_unique) & (i != ''):
        s_unique.append(i)

s_unique

#dictionary 로 껀수(빈도수) 세기 
dic = dict()
dic

for i in s:
    if i in dic:
        dic[i] += 1
    elif i != '':
        dic[i] = 1

dic.keys()
dic.values()
dic.items()

#Series로 바꾸기, 이때 공백문자를 null로 
#list 내장객체에서 for 앞에 if 쓰면 변경해서 출력하기
#for 뒤에 if 쓰면 걸러내기 
lst = [np.nan if i == '' else i for i in s]

Series(lst).unique()
Series(lst).value_counts(dropna=False)

s_unique = Series(lst).unique()
s_unique[pd.notnull(s_unique)]
type(s_unique)
s_unique.dropna()#error
#dropna 는 Series나 DataFrame에서만 된다.
Series(s_unique).dropna()