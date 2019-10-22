# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:42:39 2019

@author: stu12
"""
#10/21#
import pandas as pd

mail = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\mail.csv",
                   encoding = 'cp949')

#10/22#
#cmd에 들어가서 pip install konlpy
from konlpy.tag import Twitter

t = Twitter()
#t.pos()#여기에 문장을 넣으면 형태소 분석을 한다. 지금은 아무것도 없어서 error

text = t.pos('아버지가방에들어가신다',norm = True, stem = True)
text
#[('아버지', 'Noun'), ('가방', 'Noun'), ('에', 'Josa'), ('들어가다', 'Verb')]
t.pos('ㅇㅇ')
#[('ㅇㅇ', 'KoreanParticle')]

txt = '통찰력은 사물이나 현상의 원인과 결과를 이해하고 \
간파하는 능력이다. 통찰력을 얻는 좋은 방법은 독서이다.'

#명사만 추출하기
t.nouns(txt)
#형태소별로 추출하기 
t.pos(txt)

from konlpy.tag import Kkma
k = Kkma()
#문장별로 나누어준다. 
k.sentences(txt)
#명사만 추출하기
k.nouns(txt)
#형태소별로 추출하기
k.pos(txt)

from konlpy.tag import Hannanum

h = Hannanum()
#명사만 추출하기
h.nouns(txt)
#형태소별로 추출하기, 표기가 다르다. 
h.pos(txt)

#C:\ProgramData\Anaconda3\Lib\site-packages\konlpy\data\corpus\kolaw
#로 가면 헌법 텍스트가 있다.
from konlpy.corpus import kolaw
#위 폴더로 가서 텍스트 파일의 목록을 보여준다.
kolaw.fileids()
#헌법 파일을 열자
doc_ko = kolaw.open('constitution.txt').read()
doc_ko
tokens_ko = t.nouns(doc_ko)
tokens_ko

type(tokens_ko)#list

#pip install nltk
import nltk

ko = nltk.Text(tokens_ko)
type(ko)
ko.tokens
type(ko.tokens)#list
#단어의 갯수를 확인하자 
len(ko.tokens)#3882
len(set(ko.tokens))#929

#단어의 빈도수 
ko.vocab()
#상위 10개만 
ko.vocab().most_common(10)

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

plt.figure(figsize = (12,6))
ko.plot(50)
plt.show()

#취임사.txt를 불러와서 해 보자 
doc_ko1 = kolaw.open('취임사.txt').read()
tokens_ko1 = t.nouns(doc_ko1)
ko1 = nltk.Text(tokens_ko1)

plt.figure(figsize = (12,6))
ko1.plot(50)
plt.show()

#불용어 . 한글일때는 불용어 사전을 직접 만들어서 한다
stopword = ['.','.','(',')','의','에','해','제']

ko = [word for word in ko if word not in stopword]

ko = nltk.Text(ko)
len(ko.tokens)
len(set(ko.tokens))
ko.vocab()

plt.figure(figsize = (12,6))
ko.plot(50)
plt.show()

#단어의 갯수 
ko.count("국민")
ko.concordance("국민")
ko.similar('친구')
ko.similar('탄핵')
ko1.similar('탄핵')

#텍스트 내 단어 사용 빈도와 위치를 나타내는 그래프 
#해당 어휘가 주로 어느부분에 나오는지 보여줌 
ko1.dispersion_plot(['대통령','국민'])

#pip install wordcloud
from wordcloud import WordCloud

data = ko.vocab().most_common(10)
data

wordcloud = WordCloud(font_path = "c:\\windows\\fonts\\malgun.ttf",
                      background_color = 'white',
                      width = 1000, height = 800).\
                      generate_from_frequencies(dict(data))

plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#취임사.txt
with open("C:\\WorkSpace\\Python_Space\\data\\취임사.txt") as file:
    text = file.read()

###
from scipy.misc import imread#scipy 1.3부터 안됨 1.2.1로 다운그레이드

#텍스트 이미지들이 모여서 하트모양을 이루도록 하자 
heart_mask = imread('C:\\WorkSpace\\Python_Space\\data\\heart.jpg',
                    flatten = True)
heart_mask

wordcloud = WordCloud(font_path = "c:\\windows\\fonts\\malgun.ttf",
                      background_color = 'white',
                      width = 1000, height = 800, mask = heart_mask).\
                      generate_from_frequencies(dict(data))
                      
plt.imshow(wordcloud)
plt.show()

data = ko.vocab().most_common(10)
wordcloud = WordCloud(font_path = "c:\\windows\\fonts\\malgun.ttf",
                      background_color = 'white',
                      width = 1000, height = 800, mask = heart_mask).\
                      generate_from_frequencies(dict(data))
                      
plt.imshow(wordcloud)
plt.show()

#
import imageio
heart_mask = imageio.imread("C:\\WorkSpace\\Python_Space\\data\\heart.jpg")
wordcloud = WordCloud(font_path = "c:\\windows\\fonts\\malgun.ttf",
                      background_color = 'white',
                      width = 1000, height = 800, mask = heart_mask).\
                      generate_from_frequencies(dict(data))
plt.imshow(wordcloud)
plt.show()

#단어들을 모아서 naive bayes에 학습하고 결과를 보자 
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import nltk

#훈련시킬 (문장,라벨) 들이다.
train = [('홍길동은 좋아','긍정'),
         ('강아지는 무지 좋아','긍정'),
         ('수업이 재미없어','부정'),
         ('홍길동은 이쁜 강아지야','긍정'),
         ('민형이는 너무 멋있어','긍정'),
         ('은진이는 무지 무지 많이 이뻐','긍정'),
         ('난 수업을 마치고 희주랑 클럽갈꺼야','긍정'),
         ('오늘 하루는 너무 짜증스러운 날이야','부정'),
         ('날이 맑아서 좋아','긍정'),
         ('비가 오니 짜증난다','부정'),
         ('지하철에서 질서가 너무 없어 짜증이 난다.','부정'),
         ('공기가 맑아서 좋다','긍정'),
         ('밝게 인사해주니 행복하다','긍정'),
         ('친구가 짜증을 낸다','부정'),
         ('머신러닝 수업을 들으니 재미있어','긍정'),
         ('내일은 쉬는 날이야','긍정'),
         ('한조 대기 중','부정'),
         ('겐지가 함께한다','부정')]

#단어를 분리함
word_tokenize('홍길동은 좋아')
#['홍길동은', '좋아']

#문장을 단어별로 추출하기 
[j for i in train for j in word_tokenize(i[0])]

for i in train:
    print(i)
    print(word_tokenize(i[0]))

#train에 있는 문장에 있는 단어들을 전부 모으자 
allword = set([j for i in train for j in word_tokenize(i[0])])
allword

#해당 단어가 문장에 포함되는가?
for i in allword:
    for j in train:
        print("'{}'는 '{}'에 포함되는가? ".format(i,j[0]))
        print(bool(i in word_tokenize(j[0])))
        
#list 내장 객체로 만들면 
[({word : (word in word_tokenize(x[0])) for word in allword},
   x[1]) for x in train]
len(train)
len(allword)
true_false = [({word : (word in word_tokenize(x[0])) for word in allword},
                x[1]) for x in train]
print(true_false[0])

import nltk
model = nltk.NaiveBayesClassifier.train(true_false)
model
#주요 단어별로 True or False에 따라서 부정:긍정 비율이 나온다.
model.show_most_informative_features()

##이제 다음 단어들을 넣어봐서 결과가 어떻게 나오는지 보자 
test = '홍길동은 수업이 끝나고 클럽에 간다.'
t_t_f = {word : (word in word_tokenize(test)) for word in allword}
t_t_f
model.classify(t_t_f)

test = '날이 우울해서 짜증이 난다'
t_t_f = {word : (word in word_tokenize(test)) for word in allword}
t_t_f
model.classify(t_t_f)

text = '머신러닝 수업은 재미있다.'
t_t_f = {word : (word in word_tokenize(test)) for word in allword}
t_t_f
model.classify(t_t_f)

#이번에는 단어들을 품사별로도 구분해서 해 보자 
test = '홍길동이랑 놀꺼야'
['/'.join(t) for t in t.pos(test, norm = True, stem = True)]
#['홍길동/Noun', '이랑/Josa', '놀다/Verb']

#다음과 같이 단어/품사의 list를 반환하는 함수를 만들자
def tokenize(x):
   
    return ['/'.join(t) for t in t.pos(x, norm = True, stem = True)]

#trian에 있는 문장의 단어들을 단어/품사 형태로 반환하자 
train_doc = [(tokenize(row[0]),row[1]) for row in train]

#단어/품사 들을 한 list안에 들어가게 하자 
tokens = [j for i in train_doc for j in i[0]]
#set으로 만들어서 중복된 단어를 제거하자 
tokens = set(tokens)
len(tokens)

#tokens를 기준으로 true false을 만들자
#d는 단어, c는 긍정/부정
[(d,c) for d,c in train_doc]

#tokens에 있는 단어가 doc에 있는가 
def term_exists(doc):
    
    return {word : (word in doc) for word in tokens}

train_x = [(term_exists(d),c) for d,c in train_doc]
train_x

#모델을 훈련하자 
model = nltk.NaiveBayesClassifier.train(train_x)
#
model.most_informative_features()
test = '홍길동이랑 놀꺼야'
test_doc = tokenize(test)
#test도 train 형태이여야 함 
test_f = term_exists(test_doc)
model.classify(test_f)

def naive_bayes_test(x):
    
    x_doc = tokenize(x)
    x_f = term_exists(x_doc)
    print(model.classify(x_f))

naive_bayes_test('오늘은 짜장면을 먹어서 짜증난다.')  
naive_bayes_test('한조 대기중')

### 이제 R에서 했던 것을 (영화 게시판/리뷰) 여기서 해 보자  ###

from bs4 import BeautifulSoup
import urllib.request as req
url = "https://movie.daum.net/moviedb/grade?\
movieId=95547&type=netizen&page="

import re
score_text = []
review_text = []
for i in range(1,91):
    
    res = req.urlopen(url)

    soup = BeautifulSoup(res, "html.parser")

    #평점
    score = soup.find_all('em',{'class':'emph_grade'})

    for i in score:
    
        point = int(i.string)
        label = ''
        if point > 6:
            label = '긍정'
        else:
            label = '부정'
     
        score_text.append(label)
    

    #리뷰 
    review = soup.find_all('p',{'class':'desc_review'})

    for i in review:
        temp = i.get_text().strip()
        temp = re.sub('[\.\,\?]','',temp)
        
        review_text.append(temp)

#(문장, 라벨) 형태로 만들자 
train_data = []
for i in range(len(score_text)):
    train_data.append((review_text[i],score_text[i]))
        
train_data
len(train_data)

#train_data에 있는 문장에 있는 단어들을 전부 모으자 
allword_data = set([j for i in train_data for j in word_tokenize(i[0])])
#단어가 문장에 포함여부
true_false_data = [({word : (word in word_tokenize(x[0]))
                     for word in allword_data},
                    x[1]) for x in train_data]
    
model = nltk.NaiveBayesClassifier.train(true_false_data)
model.show_most_informative_features()

test = '이게 영화냐'
t_t_f = {word : (word in word_tokenize(test))
             for word in allword_data}
t_t_f
model.classify(t_t_f)

#########
#trian_data에 있는 문장의 단어들을 단어/품사 형태로 반환하자 
train_doc_data = [(tokenize(row[0]),row[1]) for row in train_data]

#단어/품사 들을 한 list안에 들어가게 하자 
tokens_data = [j for i in train_doc_data for j in i[0]]
#set으로 만들어서 중복된 단어를 제거하자 
tokens_data = set(tokens_data)

train_x_data = [(term_exists(d),c) for d,c in train_doc_data]

#모델을 훈련하자 
model = nltk.NaiveBayesClassifier.train(train_x_data)
#
model.most_informative_features()

naive_bayes_test('이게 영화냐')  
naive_bayes_test('스타워즈 너무 재미있다')
