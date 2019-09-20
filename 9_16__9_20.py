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
#- R의 백터(vector) 자료형하고 유사하다. 

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
#전치 transpose
df5 = df4.T
df5
df5.dtypes

df4.dtypes
df4['서울'].dtype
df4.info
df4.info()

#index를 지정해서 Series를 만들때 row가 반드시 index 순서대로 만들어 지지 않는다.
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
#reindex : index를 변경, 추가, 삭제한다. 
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
df.loc['one','x']#error
df.iloc[1,'one']#error
df.ix[1,'one']
df.ix[1,['one','two']]
df.ix[['x','y'],['one','two']]
df.loc['x']['one']
df.loc[['x','y']]['one']
df.loc['one'][['x','y']]#error
df.iloc[1][1:]
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
#      PYTHON   R  SQL
#2014      60  90   50
#2015      80  65   75
#2016      70  75   85
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

#선생님 풀이 
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
type(df['a'])#pandas.core.series.Series
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

#####################################################################
#9/19#
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from numpy import nan as NA
#[문제 125] commission_pct가  null인 사원들의 급여의 총합을 출력하세요

#1. pandas 이용해서 해결
emp = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\emp.csv')
emp['COMMISSION_PCT']
emp['COMMISSION_PCT'].isnull()
emp.loc[emp['COMMISSION_PCT'].isnull()]['SALARY'].sum()

#2. pandas 사용하지 않고 해결
import csv
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)

sum_salary = 0
for i in emp_csv:
    if i[8] == '':
        sum_salary += int(i[7])
print(sum_salary)

file.close()

#[문제 126] 부서별 급여 총액 구하세요

emp[emp['DEPARTMENT_ID'] == 20]['SALARY'].sum()
dept_list = emp['DEPARTMENT_ID'].unique()
#dept_list.dropna() Series에서만 됨, Series가 아니면 na가 0이 된다.
dept_list2 = dept_list[pd.notnull(dept_list)]

for i in dept_list2:
    dept_sum = emp['SALARY'][emp['DEPARTMENT_ID'] == i].sum()
    print('{}번 부서 급여 총액 : {}'.format(int(i),dept_sum))
#
dept_list3 = Series(dept_list)
dept_list3 = dept_list3.dropna()        

for i in dept_list3.sort_values():
    dept_sum = emp['SALARY'][emp['DEPARTMENT_ID'] == i].sum()
    print('{}번 부서 급여 총액 : {}'.format(int(i),dept_sum))
    
#소속부서가 없는 사원도 계산하여 출력하려면 ? 
emp['SALARY'][emp['DEPARTMENT_ID'].isnull()].sum()
#na가 있으면 data type이 float 이다. int로 형변환이 안 된다.
dept_list4 = Series(dept_list)
np.isnan(dept_list4)


for i in dept_list4.sort_values():
    if np.isnan(i):
        dept_sum = emp['SALARY'][emp['DEPARTMENT_ID'].isnull()].sum()
        print('{}번 부서 급여 총액 : {}'.format(i,dept_sum))
    else:
        dept_sum = emp['SALARY'][emp['DEPARTMENT_ID'] == i].sum()
        print('{}번 부서 급여 총액 : {}'.format(int(i),dept_sum))
 
# isnull을 사용해도 되지 않을까?  
dept_list4.isnull()
dept_list4.notnull()
       
for i in dept_list4.sort_values():
    if i.isnull():#이걸로는 error 
        dept_sum = emp['SALARY'][emp['DEPARTMENT_ID'].isnull()].sum()
        print('{}번 부서 급여 총액 : {}'.format(i,dept_sum))
    else:
        dept_sum = emp['SALARY'][emp['DEPARTMENT_ID'] == i].sum()
        print('{}번 부서 급여 총액 : {}'.format(int(i),dept_sum))
 
#isnull() 이 안되는 이유는 무엇일까?       
x = Series([1,2,3,np.nan, None])
x
x.isnull()
np.isnan(x)
#isnull() 은 Series용이기 때문이다.
x[0].isnull()#error
np.isnan(x[0])
np.isnan(x[4])

#list를 만들어서 해 보자 
y = [1,2,3,np.nan,None]
type(y)        
y.isnull()
np.isnan(y)#error
np.isnan(y[2])
np.isnan(y[3])
np.isnan(y[4])#error, None은 pandas에서만 제공해서 

#pandas에서 제공하는 함수를 사용하자 
pd.isna(x[4])
pd.isna(x)
pd.isnull(x)
pd.notnull(x)

#group by 기능을 가진 함수를 사용하자 
emp['SALARY'].groupby(emp['DEPARTMENT_ID']).sum()
emp['SALARY'].groupby(emp['DEPARTMENT_ID']).mean()
emp['SALARY'].groupby(emp['DEPARTMENT_ID']).max()
emp['SALARY'].groupby(emp['DEPARTMENT_ID']).min()
emp['SALARY'].groupby(emp['DEPARTMENT_ID']).count()

#select department_id, job_id, sum(salary)
#from employees
#group by department_id, job_id
#처럼 하려면 
emp['SALARY'].groupby([emp['DEPARTMENT_ID'],emp['JOB_ID']]).sum()

#sql에서 pivot를 했었다.
emp['SALARY'].groupby([emp['DEPARTMENT_ID'],emp['JOB_ID']]).sum().unstack()
#select *
#from
#(select department_id, job_id, salary
#from employees)
#pivot(sum(salary) for job_id in ('AC_ACCOUNT','AC_MGR',...,'ST_CLERK','ST_MAN'));

#... 없에려면
import pandas as pd
pd.set_option('display.max_columns',100)
#그럼 중간에  ... 없이 모든 column 들이 모두 나타난다.

#너비 확장
pd.set_option('display.width',1000)

pd.set_option('display.max_rows',1000)

#pd.set_option('display.height',1000)

#다른 group by 
emp.groupby('DEPARTMENT_ID')['SALARY'].sum()
#select department_id, sum(salary)
#from employees
#group by department_id;
emp.groupby(['DEPARTMENT_ID','JOB_ID'])['SALARY'].sum()
#select department_id, job_id, sum(salary)
#from employees
#group by department_id, job_id;
emp.groupby(['DEPARTMENT_ID','JOB_ID'])['SALARY'].sum().unstack()
#select *
#from
#(select department_id, job_id, salary
#from employees)
#pivot(sum(salary) for job_id in ('AC_ACCOUNT','AC_MGR',...,'ST_CLERK','ST_MAN'));

emp['SALARY'].groupby([emp['DEPARTMENT_ID'].fillna(0), emp['JOB_ID']]).sum()

emp.fillna(0)

emp.groupby('DEPARTMENT_ID')

for name, group in emp.groupby('DEPARTMENT_ID'):
    print(name)
    print(group)
    
for (name1, name2), group in emp.groupby(['DEPARTMENT_ID','JOB_ID']):
    print(name1, name2)
    print(group)
    
for (name1, name2), group in emp[['LAST_NAME',
    'DEPARTMENT_ID','JOB_ID']].groupby(['DEPARTMENT_ID','JOB_ID']):
    print(name1, name2)
    print(group)
    
emp.groupby('DEPARTMENT_ID')['SALARY'].sum()
emp.groupby('DEPARTMENT_ID')['SALARY'].mean()
emp.groupby('DEPARTMENT_ID')['SALARY'].aggregate(['sum','mean'])
emp.groupby('DEPARTMENT_ID').aggregate({'SALARY':'sum',
           'HIRE_DATE':['max','min']})
    
#Header가 없는 csv 파일을 읽어들여보자
emp = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\emp_new.csv',
            names = ['empid','name','job','mgr','hire_date',
                     'sal','comm','deptno'])
dept = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\dept_new.csv',
                   names = ['deptno','dname','mgr','loc'])

emp.shape
emp.columns
emp.dtypes
#deptno      int64
#dname      object
#mgr       float64
#loc         int64
#dtype: object
dept.dtypes
#empid          int64
#name          object
#job           object
#mgr          float64
#hire_date     object
#sal          float64
#comm         float64
#deptno       float64
#dtype: object

#두 DataFrame을 join하고 싶다. 
#(sql에서는 join과 merge가 다르지만 여기서는 같은 의미이다.)
pd.merge(emp, dept, on = 'deptno')
#join할 column만 골라서 해 보자 
pd.merge(emp[['name','deptno']],dept[['deptno','dname']], on = 'deptno')
#두 DataFrame의 deptno의 deta type이 다른데 작동한다...?
#데이터 형이 다르면 작동하지 않는 경우가 있다.
emp['deptno'].astype('float')

#두 column이 다른경우에는 left_no, right_no 를 사용한다.
pd.merge(emp[['name','deptno']],dept[['deptno','dname']],
         left_on = 'deptno', right_on = 'deptno')
#select e.empid, e.deptno, d.dname
#from employees e, departments d
#where e.deptno = d.deptno;

#기본 join (inner join, simple join, equi join, 등가조인)
pd.merge(emp[['name','deptno']],dept[['deptno','dname']],
         on = 'deptno', how = 'inner')

#left outer join
pd.merge(emp[['name','deptno']],dept[['deptno','dname']],
         on = 'deptno', how = 'left')
#select e.empid, e.deptno, d.dname
#from employees e, departments d
#where e.deptno = d.deptno(+);
#select e.empid, e.deptno, d.dname
#from employees e left outer join departments d
#where e.deptno = d.deptno;

#right outer join
pd.merge(emp[['name','deptno']],dept[['deptno','dname']],
         on = 'deptno', how = 'right')
#select e.empid, e.deptno, d.dname
#from employees e, departments d
#where e.deptno(+) = d.deptno;
#select e.empid, e.deptno, d.dname
#from employees e right outer join departments d
#where e.deptno = d.deptno;

#full outer join
pd.merge(emp[['name','deptno']],dept[['deptno','dname']],
         on = 'deptno', how = 'outer')
#select e.empid, e.deptno, d.dname
#from employees e full outer join departments d
#where e.deptno = d.deptno;

dept = DataFrame({'dname':['관리팀','마케팅팀','구매팀','인사팀',
                           '경영지원팀','기술지원팀','홍보팀','기획팀',
                           '재무팀','회계팀']},
                    index = [10,20,30,40,50,60,70,80,90,100])   
dept
#하나는 column, 하나는 index를 기준으로 join한다.
pd.merge(emp[['name','deptno']], dept,
         left_on = 'deptno', right_index = True)

df1 = DataFrame([[54,34,87],[95,34,76],[92,86,73],[8,444,54]],
                index = ['홍길동','박찬호','손흥민','한조'],
                columns = ['영어','수학','국어'])
df1
df2 = DataFrame([[54,34,87],[95,34,76],[92,86,73],[66,55,30]],
                index = ['홍길동','박찬호','손흥민','겐지'],
                columns = ['과학','음악','체육'])
df2
#index를 기준으로 join한다.
pd.merge(df1, df2, left_index = True, right_index = True)
#     영어  수학  국어  과학  음악  체육
#홍길동  54  34  87  54  34  87
#박찬호  95  34  76  95  34  76
#손흥민  92  86  73  92  86  73
#한조는 안 나온다. left outer join을 해서 나오게 하자 
pd.merge(df1, df2, left_index = True, right_index = True, how = 'left')
#     영어   수학  국어    과학    음악    체육
#홍길동  54   34  87  54.0  34.0  87.0
#박찬호  95   34  76  95.0  34.0  76.0
#손흥민  92   86  73  92.0  86.0  73.0
#한조    8  444  54   NaN   NaN   NaN

df1.join(df2, how = 'inner')
df1.join(df2, how = 'outer')
df1.join(df2)#left outer join

#[문제 127] emp에서 job이 AD_VP, AD_PRES인 사원들의 이름, 급여, 부서코드,
#부서이름을 출력해주세요 
table1 = pd.merge(emp[['name','job','sal','deptno']],
         dept, left_on = 'deptno', right_index = True)

table1
type(table1)
table1['job'].isin(['AD_VP','AD_PRES'])
table1[table1['job'].isin(['AD_VP','AD_PRES'])]

table2 = pd.merge(emp[emp['job'].isin(['AD_VP','AD_PRES'])]
                      [['job','name','sal','deptno']],dept,
                    left_on = 'deptno', right_index = True)
table2

#[문제 128] 부서이름별 급여 총액을 출력해주세요
dept_list = table1['dname'].unique()
dept_list

for i in dept_list:
    sum_sal = table1['sal'][table1['dname'] == i].sum()
    print('{} 총 급여 : {}'.format(i,sum_sal))
    
dept_sal = emp['sal'].groupby(emp['deptno']).sum()
#merge하기 전에 DataFrame으로 바꾸자 
dept_sal = DataFrame(dept_sal)

pd.merge(dept_sal, dept, left_on = 'deptno', 
         right_index = True)[['dname','sal']]

dept_sal2 = emp['sal'].groupby(emp['deptno'].fillna(0)).sum()
#merge하기 전에 DataFrame으로 바꾸자 
dept_sal2 = DataFrame(dept_sal2)

pd.merge(dept_sal2, dept, left_on = 'deptno', 
         right_index = True, how='right')[['dname','sal']]

#[문제 129] 50번 부서 사원중에 급여가 5000 이상인 사원이름, 부서이름을 출력하세요
emp_50 = emp[(emp['sal'] >= 5000) & (emp['deptno'] == 50)][['name','deptno']]
pd.merge(emp_50, dept, left_on = 'deptno', right_index = True)[['name','dname']]
pd.merge(emp_50, dept[['deptno','dname']], on = 'deptno')[['name','dname']]


#2개의 DataFrame을 불러들이자 
df1 = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\emp1.csv',
            names = ['empid','name','hire_date','sa','deptno'])
df1
df2 = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\emp2.csv',
            names = ['empid','name','hire_date','sa','deptno'])
df2

#2개의 DataFrame을 한 DataFrame에 넣어보자 
df3 = pd.DataFrame()
for i in range(1,3):
    file = 'C:\\WorkSpace\\Python_Space\\data\\emp{}.csv'.format(i)
    temp = pd.read_csv(file,names = ['empid','name','hire_date','sa','deptno'])
    df3 = df3.append(temp)
df3

df3.groupby('deptno').aggregate({'sa':'sum','hire_date':['max','min']})
#생각해보면  emp 뒤에 3,6,8 등 하면 반복문으로 하기 힘들다.

#새로운 폴더를 만들어 놓고 
import glob
file = 'C:\\WorkSpace\\Python_Space\\csv\\emp*.csv'
#해당 폴더에 가서 emp로 시작하는 csv파일명들의 목록을 만든다.
file_lst = glob.glob(file)

#아까 반복문을 수정해서 작동해보자 
df4 = pd.DataFrame()
for i in file_lst:
    temp = pd.read_csv(i,names = ['empid','name','hire_date','sa','deptno'])
    df4 = df4.append(temp)
df4

#############################################################
#9/20#
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from numpy import nan as NA
#[문제 130] 2010년도에 태어난 아이 이름, 성별 상관없이 상위 10명 출력하세요.

year2010 = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\year2010.csv',
                       names=['name','gender','number'])
year2010
type(year2010)
year2010.dtypes
year2010.info()

#rank 사용
year2010['number'].rank(ascending=False, method='dense')
year2010['number'].rank(ascending=False, method='first')
year2010[year2010['number'].rank(ascending=False, method='first') <= 10]
#            name gender  number
#0       Isabella      F   22731
#1         Sophia      F   20477
#2           Emma      F   17179
#3         Olivia      F   16860
#19698      Jacob      M   21875
#19699      Ethan      M   17866
#19700    Michael      M   17133
#19701     Jayden      M   17030
#19702    William      M   16870
#19703  Alexander      M   16634

#sort_values사용
rank10 = year2010['number'].sort_values(ascending = False)[0:10]
type(rank10)#Series
rank10.values[-1]
year2010[year2010['number'] >= rank10.values[-1]]

#column 추가 
year2010['rankFirst'] = year2010['number'].rank(ascending=False, method='first')
year2010[year2010['rankFirst'] <= 10]
#            name gender  number  rankFirst
#0       Isabella      F   22731        1.0
#1         Sophia      F   20477        3.0
#2           Emma      F   17179        5.0
#3         Olivia      F   16860        9.0
#19698      Jacob      M   21875        2.0
#19699      Ethan      M   17866        4.0
#19700    Michael      M   17133        6.0
#19701     Jayden      M   17030        7.0
#19702    William      M   16870        8.0
#19703  Alexander      M   16634       10.0

#[문제 131] 2010년도에 태어난 아이 이름, 성별 따로 상위 5명 출력하세요.
year2010 = year2010.loc[:,['name','gender','number']]

#rank 사용
year2010['gender'] == 'M'
year2010M = year2010[year2010['gender'] == 'M']
year2010F = year2010[year2010['gender'] == 'F']
year2010M[year2010M['number'].rank(ascending=False, method='first') <= 5]
year2010F[year2010F['number'].rank(ascending=False, method='first') <= 5]

#sort_values 사용
year2010[year2010['gender'] == 'M'].sort_values(
        by = 'number',ascending = False)[:5]
year2010[year2010['gender'] == 'F'].sort_values(
        by = 'number',ascending = False)[:5]

#한 DataFrame에서 출력할 방법은? groupby를 이용하자 
year2010.groupby('gender')['number'].rank(ascending=False, method='first')
year2010_gender_10 = year2010[year2010.groupby('gender')['number'].rank(
        ascending=False, method='first') <= 5]

year2010_gender_10
#           name gender  number
#0      Isabella      F   22731
#1        Sophia      F   20477
#2          Emma      F   17179
#3        Olivia      F   16860
#4           Ava      F   15300
#19698     Jacob      M   21875
#19699     Ethan      M   17866
#19700   Michael      M   17133
#19701    Jayden      M   17030
#19702   William      M   16870
    
#[문제 132] 2010년도에 태어난 아이 이름, 성별 상관없이 상위 10위까지 출력하세요.
#rank 사용
year2010[year2010['number'].rank(ascending=False, method='dense') <= 10]
#            name gender  number
#0       Isabella      F   22731
#1         Sophia      F   20477
#2           Emma      F   17179
#3         Olivia      F   16860
#19698      Jacob      M   21875
#19699      Ethan      M   17866
#19700    Michael      M   17133
#19701     Jayden      M   17030
#19702    William      M   16870
#19703  Alexander      M   16634

#sort_values 사용
year2010.sort_values(by='number', ascending = False)[:10]
#            name gender  number
#0       Isabella      F   22731
#19698      Jacob      M   21875
#1         Sophia      F   20477
#19699      Ethan      M   17866
#2           Emma      F   17179
#19700    Michael      M   17133
#19701     Jayden      M   17030
#19702    William      M   16870
#3         Olivia      F   16860
#19703  Alexander      M   16634

#column 을 추가
year2010['rankDense'] = year2010['number'].rank(ascending=False, method='dense')
year2010[year2010['rankDense'] <= 10]
#            name gender  number  rankDense
#0       Isabella      F   22731        1.0
#1         Sophia      F   20477        3.0
#2           Emma      F   17179        5.0
#3         Olivia      F   16860        9.0
#19698      Jacob      M   21875        2.0
#19699      Ethan      M   17866        4.0
#19700    Michael      M   17133        6.0
#19701     Jayden      M   17030        7.0
#19702    William      M   16870        8.0
#19703  Alexander      M   16634       10.0

#[문제 133] 2010년도에 태어난 아이 이름, 성별 따로 상위 5위까지 출력하세요.
year2010 = year2010.loc[:,['name','gender','number']]

year2010M[year2010M['number'].rank(ascending=False, method='dense') <= 5]
year2010F[year2010F['number'].rank(ascending=False, method='dense') <= 5]

#groupby를 이용
year2010.groupby('gender')['number'].rank(ascending=False, method='dense')
year2010_gender_10 = year2010[year2010.groupby('gender')['number'].rank(
        ascending=False, method='dense') <= 5]

#상위 10명이나 상위 10위나 여기서는 결과가 같다. 중복되는 값이 없기 때문이다.

#[문제 134] 2001년 ~ 2016년도 년도별 출생수 
import glob
file = 'C:\\WorkSpace\\Python_Space\\csv\\yob*.csv'
file_lst = glob.glob(file)
file_lst[0][-11:-4]

df_yob = DataFrame()
year = 2000
year_birth = []
for i in file_lst:
    temp = DataFrame()
    temp = pd.read_csv(i, names = ['name','gender','number'])
    temp['year'] = year
    df_yob = df_yob.append(temp)
    year_birth.append([year, temp['number'].sum()])
    year += 1

df_yob
df_yob.info()
year_birth


#선생님 풀이, 출력하며 저장해보자 
with open('C:\\WorkSpace\\Python_Space\\data\\year.txt','w') as f:
    for x, y in year_birth:
        data = '%s, %s\n'%(x,y)
        print(data)
        f.write(data)

#년도별 출생자수 
df_yob.groupby(['year'])['number'].sum()
#년도와 성별 출생자수
df_yob.groupby(['year','gender'])['number'].sum()
df_yob.groupby(['year','gender'])['number'].sum().unstack()


df_yob.groupby(['year']).aggregate({'number':['sum']})

#[문제 135] 2000년 ~ 2016년 태어난 아이 이름 성별 상관없이 상위 10위까지 출력하세요
df_yob['number'].rank(ascending = False)
df_yob[df_yob['number'].rank(ascending = False) <= 10]
df_yob[df_yob['number'].rank(ascending = False, method = 'dense') <= 10]
df_yob[df_yob['number'].rank(ascending = False, method = 'mean') <= 10]
##
#년도별 합쳐서 이름을 성별에 상관없이 10위까지 
df_yob.groupby('name')['number'].sum()
rankName = df_yob.groupby('name')['number'].sum().rank(
        ascending = False, method = 'dense')
pd.merge(df_yob, rankName, on = 'name')

df_yob.groupby(['name','gender'])['number'].sum()
name_gender = df_yob.groupby(['name','gender'])['number'].sum()
name_gender[name_gender.rank(ascending = False, method = 'dense') <= 10]

#[문제 136] 2000년 ~ 2016년 태어난 아이 이름 성별 따로 상위 5위까지 출력하세요
year2010 = year2010.loc[:,['name','gender','number']]
df_yob_sum = df_yob.groupby(['name','gender'])['number'].sum()
df_yob_sum
type(df_yob_sum)
df_yob_sum[df_yob_sum.rank(ascending = False, method = 'dense') <= 10]
#name     gender
#Andrew   M         284683
#Daniel   M         303484
#Emily    F         314233
#Emma     F         319903
#Ethan    M         314794
#Jacob    M         400578
#Joshua   M         315248
#Matthew  M         315179
#Michael  M         360013
#William  M         310448
#Name: number, dtype: int64


## 선생님의 풀이 ##
file = 'C:\\WorkSpace\\Python_Space\\csv\\yob*.csv'
file_lst = glob.glob(file)
#우선 2000년도와 2001년도를 marge 해 보자 
df = pd.read_csv('C:\\WorkSpace\\Python_Space\\csv\\yob2000.csv',
                 names = ['name','gender','birth'])
df

year = pd.read_csv('C:\\WorkSpace\\Python_Space\\csv\\yob2001.csv',
                   names = ['name','gender','birth'])
year
#위에 2개를 merge를 해 보자, 2000년도 출생수는 birth_x column에
#2001년도 출생수는 birth_y column에 생긴다.
x = pd.merge(df,year,on=['name','gender'], how = 'outer')
x
x.iloc[:,2]#2000년도 출생수
x.iloc[:,3]#2001년도 출생수
#birth column에 두 column의 수를 더하자 
x['birth'] = x.iloc[:,2].add(x.iloc[:,3], fill_value = 0)
x
#이제 name, gender, birth column만 꺼내어 df를 구성하자 
df = x.loc[:,['name','gender','birth']]
df

#그러면 위와 같은 과정을 2000년 부터 2016년도 까지 for문을 이용해서 반복하면 된다.
df = pd.read_csv('C:\\WorkSpace\\Python_Space\\csv\\yob2000.csv',
                 names = ['name','gender','birth'])

for i in file_lst[1:]:
    year = pd.read_csv(i, names = ['name','gender','birth'])
    x = pd.merge(df,year,on=['name','gender'], how = 'outer')
    x['birth'] = x.iloc[:,2].add(x.iloc[:,3], fill_value = 0)
    df = x.loc[:,['name','gender','birth']]
    
df
df['rank'] = df['birth'].rank(ascending = False, method = 'dense')
df[df['rank'] <= 10].sort_values('rank') 
#          name gender     birth  rank
#17653    Jacob      M  400578.0   1.0
#17654  Michael      M  360013.0   2.0
#16        Emma      F  319903.0   3.0
#17656   Joshua      M  315248.0   4.0
#17655  Matthew      M  315179.0   5.0
#17677    Ethan      M  314794.0   6.0
#0        Emily      F  314233.0   7.0
#17663  William      M  310448.0   8.0
#17661   Daniel      M  303484.0   9.0
#17659   Andrew      M  284683.0  10.0


#[문제 137] 2000년 ~ 2016년 성별 출생현황을 출력해 주세요 
df_yob.groupby(['gender','year'])['number'].sum().unstack()
df_yob.groupby(['year','gender'])['number'].sum().unstack()
df_yob_y_g = df_yob.groupby(['year','gender'])['number'].sum().unstack()
df_yob_y_g.dtypes
df_yob_y_g.info()
df_yob_y_g.iloc[0,:]
df_yob_y_g.index[0]

list1 = []
for i in range(len(df_yob_y_g)):
    x,y = df_yob_y_g.iloc[i,:]
    z = df_yob_y_g.index[i]
    print(z,x,y)
    list1.append([z,x,y])
list1

for i in list1:
    print('{}년도 여자 : {}명, 남자 : {}명'.format(i[0],i[1],i[2]))

df_yob.groupby(['gender'])['number'].sum()
df_yob.groupby(['year'])['number'].sum()

#선생님의 풀이
file = 'C:\\WorkSpace\\Python_Space\\csv\\yob*.csv'
file_lst = glob.glob(file)
year = 2000

with open('C:\\WorkSpace\\Python_Space\\data\\year_gender_total.txt',
          'w') as f:#encoding = 'utf-8'
    f.write('{},{},{}\n'.format('년도','여자','남자'))
    #f.write에서 빈칸 띄우면 다른 column name 이 된다.... 
    for i in file_lst:
        df = pd.read_csv(i, names = ['name','gender','birth'])
        gender_cn = df['birth'].groupby(df['gender']).sum()
        data = '%s,%s,%s\n'%(year, gender_cn.loc['F'],
                             gender_cn.loc['M'])
        f.write(data)
        year += 1

df = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\year_gender_total.txt') 
df = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\year_gender_total.txt',
                 encoding = 'CP949')
df.info() 
df.columns
df['여자']
df['남자'].sum()

#다른 풀이, import os 를 해서 ~ 
import csv
import os

#파일경로중에서 파일이름.확장자 를 뽑아낸다.
os.path.basename('C:\\WorkSpace\\Python_Space\\csv\\yob2000.csv')#'yob2000.csv'

'yob2000.csv'.split('.')#['yob2000', 'csv']
'yob2000.csv'.split('.')[0]#'yob2000'

with open('C:\\WorkSpace\\Python_Space\\data\\year_gender_total.txt',
          'w',
          newline='',
          encoding='utf-8') as f:
    #newline 없으면 빈 줄이 있는 상태로 저장된다.
    writer = csv.writer(f, delimiter = ',')#열을 ,로 나눈다.
    writer.writerow(['년도','여자','남자'])#행을 입력한다.
    for y in range(2000,2017):
        filename = 'C:\\WorkSpace\\Python_Space\\csv\\yob%d.csv'%y
        name = os.path.basename(filename)
        name = name.split('.')[0]
        df = pd.read_csv(filename, names = ['name','gender','birth'])
        gender_cn = df['birth'].groupby(df['gender']).sum()
        writer.writerow([name[3:], gender_cn.loc['F'], gender_cn.loc['M']])

df = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\year_gender_total.txt')
df.info() 
df.columns
df['여자']
df['남자'].sum()


