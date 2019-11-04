# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 09:56:22 2019

@author: stu12
"""

#11/4#
#[문제] 좋아 하는 영화에 대해서 긍정적 평가와 부정적 평가에 대한 
#워드 클라우드, 연관분석을 수행해보세요.
#
#1. 데이터를 로드한다.  
#2. 영화평점이 9점이상은 긍정변수넣고 2점 이하는 부정변수에 넣는다 
#3. 긍정변수에서 명사만 추출하고 데이터 정제 작업을 한다.
#4. 부정변수에서 명사만 추출하고 데이터 정제 작업을 한다.
#5. 긍정 단어와 부정단어를 각각 워드 클라우드로 그려서 한 화면에
#출력한다. 
#6  긍정적 평가 게시판의 글들을 명사만 추출한 다음 단어들간의
# 연관관계를 출력하시오
import re
from bs4 import BeautifulSoup
import urllib.request as req
from selenium import webdriver

#daum 영와 평점 게시판의 글을 수집하자 
url = 'https://movie.daum.net/moviedb/grade?\
movieId=123582&type=netizen&page={}'

score_text = []
review_text = []

#1 ~ 20 페이지의 게시판 글 수집 
for i in range(0,20):
    
    html = req.urlopen(url.format(i))

    soup  = BeautifulSoup(html, "html.parser")

    score = soup.find_all('em',{'class' : 'emph_grade'})

    for i in score:
        score_text.append(int(i.get_text()))
    
    review = soup.find_all('p',{'class':'desc_review'})

    for i in review:
        temp = i.get_text().strip()       
        review_text.append(temp)
        
    
len(score_text)#200
len(review_text)#200
type(review_text)

for j in range(len(review_text)):
    i = review_text[j]
    i = re.sub('\s+',' ',i)
    i = re.sub('터미네이터 {0,}1','터미네이터원',i)
    i = re.sub('터미네이터 {0,}2','터미네이터투',i)
    i = re.sub('터미네이터 {0,}3','터미네이터쓰리',i)
    i = re.sub('터미네이터 {0,}4','터미네이터포',i)
    i = re.sub('1편','터미네이터원',i)
    i = re.sub('2편|2부|T2|터2|터미2','터미네이터투',i)
    i = re.sub('3편|라이즈 오브 머신','터미네이터쓰리',i)
    i = re.sub('4편|미래전쟁의 시작','터미네이터포',i)
    i = re.sub('5편|제니시스|제네시스','터미네이터파이브',i)
    i = re.sub('6편|다크 {0,}페이트','터미네이터식스',i)
    i = re.sub('zz','쿨쿨',i)
    i = re.sub('T1000|T-1000','T천',i)
    i = re.sub('2019년','이천십구년',i)
    i = re.sub('1991년|91년','천구십일년',i)
    i = re.sub('아놀드 {1,}','아놀드슈알제네거',i)
    i = re.sub('슈워제네거|슈워츠네거','슈왈제네거',i)
    i = re.sub('존 {0,}코너','존코너',i)
    i = re.sub('사라 {0,}코너','사라코너',i)
    i = re.sub('재밌','재미있',i)
    i = re.sub('페미','페미니즘',i)
    i = re.sub('중2병','중이병(허세)',i)
    i = re.sub('망작','망한작품',i)
    i = re.sub('할아범','할아버지',i)
    i = re.sub('10점','십점',i)
    i = re.sub('ㅅㅂ','시발',i)
    i = re.sub('항수','향수',i)
    #i = re.sub('[0-9]+',' ',i)
    i = re.sub('[!|^|%|~|*|?|♡]+',' ',i)
    review_text[j] = i

review_text

#모은 데이터로 데이터프레임을 만들자 
from pandas import Series, DataFrame

df = DataFrame({'score' : score_text, 'review' : review_text},
               index = range(len(score_text)))
df
df.info()
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 200 entries, 0 to 199
#Data columns (total 2 columns):
#score     200 non-null int64
#review    200 non-null object
#dtypes: int64(1), object(1)
#memory usage: 3.2+ KB

#평점이 9점 이상과 2점 미만만 추출하여 새로운 데이터프레임을 만들자 
inform_df = df[(df['score'] >= 9) | (df['score'] <= 2)]
inform_df
inform_df['score'].unique()


#9점 이상을 1, 2점 이하를 0으로 바꾸자 
#1이 긍정, 0이 부정이다.
inform_df['PN'] = [1 if i >= 9 else 0 for i in inform_df['score']]
inform_df.iloc[:,[0,2]]

import collections
collections.Counter(inform_df['score'])
#Counter({9: 41, 10: 48, 1: 11, 0: 1, 2: 1})
collections.Counter(inform_df['PN'])
#Counter({1: 89, 0: 13})

#명사 추출을 해 보자 
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

#for i in inform_df.iloc[:10,1]:
#    print(i)
#
#words = set([j for i in inform_df.iloc[:100,1] for j in word_tokenize(i)])
#words
#
#PWords = set([j for i in inform_df[inform_df['PN'] == 1]['review'] 
#              for j in word_tokenize(i)])
#    
#NWords = set([j for i in inform_df[inform_df['PN'] == 0]['review'] 
#              for j in word_tokenize(i)])
#    
##불용어를 제거하자
#stopword = ['(',')','|','~']
#PWords = [i for i in PWords if i not in stopword]

#from konlpy.tag import Kkma
#
#k = Kkma()
#

from konlpy.tag import Twitter
t = Twitter()
#명사를 추출하자 

#긍정 문장의 단어
P_text_n = []
for i in inform_df[inform_df['PN'] == 1]['review']:
    #print(type(i))
    #list형태로 반환되므로 extend로 추가함 
    P_text_n.extend(t.nouns(i))
  
#부정 문장의 단어
N_text_n = []
for i in inform_df[inform_df['PN'] == 0]['review']:
    N_text_n.extend(t.nouns(i))

#출현 빈도수가 높은 단어들만 
data = collections.Counter(P_text_n).most_common(40)
data
data2 = collections.Counter(N_text_n).most_common(40)
data2

#추출한 단어들로 word cloud를 그려보자 
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

from wordcloud import WordCloud

wordcloud = WordCloud(font_path = "c:\\windows\\fonts\\malgun.ttf",
                      background_color = 'white',
                      width = 1000, height = 800).\
                      generate_from_frequencies(dict(data))

plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud2 = WordCloud(font_path = "c:\\windows\\fonts\\malgun.ttf",
                      background_color = 'white',
                      width = 1000, height = 800).\
                      generate_from_frequencies(dict(data2))

plt.imshow(wordcloud2)
plt.axis("off")
plt.show()

## navie bayes에 학습시켜보자 ##
inform_df['PN'] = inform_df.PN.map({0 : 'N', 1 : 'P'})
collections.Counter(inform_df['PN'])
#Counter({'P': 89, 'N': 13})

allword = set([j for i in range(len(inform_df)) 
for j in word_tokenize(inform_df.iloc[i,1])])

allword
           

P_N = [({word : (word in word_tokenize(inform_df.iloc[i,1])) 
                for word in allword},
                inform_df.iloc[i,2]) for i in range(len(inform_df))]
    
P_N
import nltk
model = nltk.NaiveBayesClassifier.train(P_N)
model.show_most_informative_features()

#아니면 처음부터 train_data를 list안 tuple형으로 만들고 해 보자 
train_data = []
i = 0
while(i < len(score_text)):
    #print('while')
    if (int(score_text[i]) >= 9):
        #print('p')
        train_data.append((review_text[i],'Positive'))
    elif (int(score_text[i]) <= 2):
        #print('n')
        train_data.append((review_text[i],'Negative'))
    
    i = i + 1
    #print(i)

allword2 = set([j for i in train_data for j in word_tokenize(i[0])])

collections.Counter(allword2)

P_N = [({word : word in word_tokenize(i[0]) for word in allword2},
         i[1])
       for i in train_data]

P_N
model2 = nltk.NaiveBayesClassifier.train(P_N)
model2.show_most_informative_features()
#Most Informative Features
#                   터미네이터 = True           Negati : Positi =      9.0 : 1.0
#                       등 = True           Negati : Positi =      6.4 : 1.0
#                      억지 = True           Negati : Positi =      6.4 : 1.0
#                     제임스 = True           Negati : Positi =      6.4 : 1.0
#                    망한작품 = True           Negati : Positi =      6.4 : 1.0
#                      완전 = True           Negati : Positi =      6.4 : 1.0
#                     카메론 = True           Negati : Positi =      6.4 : 1.0
#                      진짜 = True           Negati : Positi =      3.9 : 1.0
#                      가장 = True           Negati : Positi =      3.9 : 1.0
#                       이 = True           Negati : Positi =      3.9 : 1.0

test = '이게 영화냐'
t_p_n = {word : (word in word_tokenize(test))
             for word in allword2}
t_p_n
model2.classify(t_p_n)

#길이가 1이하인 단어를 제거하고 해 보자 
allword3 = [i for i in allword2 if (len(i) > 1)]

collections.Counter(allword3)

P_N2 = [({word : word in word_tokenize(i[0]) for word in allword3},
         i[1])
       for i in train_data]



model3 = nltk.NaiveBayesClassifier.train(P_N2)
model3.show_most_informative_features()
#Most Informative Features
#                   터미네이터 = True           Negati : Positi =      9.0 : 1.0
#                      억지 = True           Negati : Positi =      6.4 : 1.0
#                     제임스 = True           Negati : Positi =      6.4 : 1.0
#                    망한작품 = True           Negati : Positi =      6.4 : 1.0
#                      완전 = True           Negati : Positi =      6.4 : 1.0
#                     카메론 = True           Negati : Positi =      6.4 : 1.0
#                      진짜 = True           Negati : Positi =      3.9 : 1.0
#                      가장 = True           Negati : Positi =      3.9 : 1.0
#                      ...  = True           Negati : Positi =      3.0 : 1.0
#                      영화 = True           Negati : Positi =      2.9 : 1.0

#'터미네이터' 가 있는게 N이 P보다 왜 9배나 될까???

P_N2[0][0]['터미네이터']
P_N2[0][1]

count = {}
for i in range(len(P_N2)):
    tf = P_N2[i][0]['터미네이터']
    tf = str(tf)
    pn = P_N2[i][1]
    temp = tf + ':' + pn
    
    if temp in count.keys():
        count[temp] += 1
    else:
        count[temp] = 1
   
count

3/13
2/89
(3/13)/(2/89)

#'망한작품'에 대해서 해 보자 
count = {}
for i in range(len(P_N2)):
    tf = P_N2[i][0]['망한작품']
    tf = str(tf)
    pn = P_N2[i][1]
    temp = tf + ':' + pn
    
    if temp in count.keys():
        count[temp] += 1
    else:
        count[temp] = 1
   
count
(1/13)/(1/89)

#(값은 정확하지는 않지만) 계산한 결과로는 다음의 값에서 나온듯 하다. 
#P(부정|터미네이터 있음) : P(긍정|터미네이터 있음) = 3/13 : 2/89 = 약 10.2 : 1
#P(부정|망한작품 있음) : P(긍정|망한작품 있음) = 1/13 : 1/89 = 약 6.8 : 1
#정확하지 않는 이유는 라이브러리 내부의 알고리즘을 분석해야 알 수 있을듯 하다. 