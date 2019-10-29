# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:51:28 2019

@author: stu12
"""

########################################################################
#10/28#
import pandas as pd
#pip install mlxtend 또는 conda install mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#list모양으로 만들기 
buylist = [['우유','버터','시리얼'],
           ['우유','시리얼'],
           ['우유','빵'],
           ['버터','맥주','오징어'],
           ['커피','초콜릿'],
           ['우유','삼각김밥']]

t = TransactionEncoder()
t_ary = t.fit(buylist).transform(buylist)
print(t_ary)
#[[False  True False False  True False  True False False]
# [False False False False  True False  True False False]
# [False False  True False False False  True False False]
# [ True  True False False False  True False False False]
# [False False False False False False False  True  True]
# [False False False  True False False  True False False]]
t.columns_
#['맥주', '버터', '빵', '삼각김밥', '시리얼', '오징어', '우유', '초콜릿', '커피']

df = pd.DataFrame(t_ary, columns = t.columns_)
df
#    맥주    버터    빵 삼각김밥 시리얼  오징어  우유 초콜릿   커피
#0  False   True  False  False   True  False   True  False  False
#1  False  False  False  False   True  False   True  False  False
#2  False  False   True  False  False  False   True  False  False
#3   True   True  False  False  False   True  False  False  False
#4  False  False  False  False  False  False  False   True   True
#5  False  False  False   True  False  False   True  False  False

#지지도(support)가 0.3이 넘는것만 나오게 하자 
item = apriori(df, min_support = 0.3, use_colnames = True)
item
#    support      itemsets
#0  0.333333       (버터)
#1  0.333333      (시리얼)
#2  0.666667       (우유)
#3  0.333333  (우유, 시리얼)

a = association_rules(item)
a.info()
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 1 entries, 0 to 0
#Data columns (total 9 columns):
#antecedents           1 non-null object
#consequents           1 non-null object
#antecedent support    1 non-null float64
#consequent support    1 non-null float64
#support               1 non-null float64
#confidence            1 non-null float64
#lift                  1 non-null float64
#leverage              1 non-null float64
#conviction            1 non-null float64
#dtypes: float64(7), object(2)
#memory usage: 200.0+ bytes

#a의 column이 모두 출력이 되지 않아서 설정을 바꾸어 보자 
pd.set_option('display.max_columns',20)
print(a)
#  antecedents consequents  antecedent support  consequent support   support  \
#0    (시리얼)       (우유)           0.333333            0.666667  0.333333   
#
#   confidence  lift  leverage  conviction  
#0         1.0   1.5  0.111111         inf

#utf-8 로 인코딩 되어서 저장해놓아야 한다. 
building = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\building.csv")

#첫번째 column이 건물번호를 나타내는 것이라서 삭제해도 된다.
building = building.iloc[:,1:]

#NA를 모두 0으로 바꾸자 
building.fillna(0, inplace = True)
print(building)

#1,0을 True, False로 바꾸기 
building.astype(bool)
building = building.astype(bool)
print(building)
#그럼 column들의 이름이 다 있고, 데이터들이 모두 True, False인 
#DataFrame이 만들어진다. 이제 apriori()에 넣자 

building_item = apriori(building, min_support = 0.3, use_colnames = True)
building_item
#   support          itemsets
#0     0.30              (병원)
#1     0.30           (휴대폰매장)
#2     0.40           (일반음식점)
#3     0.45         (패밀리레스토랑)
#4     0.30             (화장품)
#5     0.40  (패밀리레스토랑, 일반음식점)

a = association_rules(building_item)
type(a)
a.columns
a.info()
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 2 entries, 0 to 1
#Data columns (total 9 columns):
#antecedents           2 non-null object
#consequents           2 non-null object
#antecedent support    2 non-null float64
#consequent support    2 non-null float64
#support               2 non-null float64
#confidence            2 non-null float64
#lift                  2 non-null float64
#leverage              2 non-null float64
#conviction            2 non-null float64
#dtypes: float64(7), object(2)
#memory usage: 272.0+ bytes


print(a)
#       antecedents            consequents   
#0   (패밀리레스토랑)          (일반음식점)         
#1     (일반음식점)          (패밀리레스토랑)         
#
#  antecedent support    consequent support  support
#0         0.45                  0.40          0.4
#1         0.40                  0.45          0.4
#   confidence      lift  leverage  conviction  
#0    0.888889  2.222222      0.22         5.4  
#1    1.000000  2.222222      0.22         inf  

#################################################################
#10/29#
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#utf-8 로 인코딩 되어서 저장해놓아야 한다.
groceries = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\groceries.csv",
                        error_bad_lines = False)
groceries#약 3000여 row가 없음 ...
groceries.info()
pd.set_option('display.max_columns',20)

groceries2 = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\groceries.csv",
                        sep = 'delimiter')

groceries2.info()

#
import csv

groceries = []
with open("C:\\WorkSpace\\Python_Space\\data\\groceries.csv","r",
          encoding = 'utf-8') as f:
    data = csv.reader(f)
    for i in data:
        print(i)
        groceries.append(i)
    
groceries
type(groceries)
groceries[0]
#list모양으로 저장이 되어 있다.
#이후로는 전날에 했던대로 하면 된다.

t = TransactionEncoder()
t_ary = t.fit(groceries).transform(groceries)
t.columns_

df = pd.DataFrame(t_ary, columns = t.columns_)
df

#min_support를 좀 작게 해야 한다. 그래야 나중에 
#association_rules를 했을때 결과가 있다. 
item = apriori(df, min_support = 0.002, use_colnames = True)
item

groceries_a = association_rules(item)
groceries_a.info()
print(groceries_a)
