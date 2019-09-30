# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:08:10 2019

@author: stu11
"""

#9/30#
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

#정규 표현식(Regular Expression)
#-특정한 패턴과 일치하는 문자열을 검색, 치환, 제거하는 기능을 제공한다. 
#-전화번호 형식, 매일 형식, 숫자로만 이루어진 문자열 

#매타문자 
#. : 문자를 의미 

import re
#re.match(찾을문자패턴, 문자열)
re.match('a','aa')
re.match('c.t','cat')
#<re.Match object; span=(0, 3), match='cat'>#문자열 0번부터 3번 이전까지 매칭
re.match('c.t','caat cat')

#. : 문자 한글자를 의미
bool(re.match('c.t','cat'))
bool(re.match('c.t','caat cat'))
re.match('c\.t','caat cat c.t')#. 이 있는경우 \.으로 찾기 
bool(re.match('c\.t','caatcatc.t'))
bool(re.match('c\.t','c.tcat'))

#* : 0번, 1번 이상 
bool(re.match('ca*t','cat'))
bool(re.match('ca*t','caat'))
bool(re.match('c*a','at time'))
bool(re.match('c*a','dat'))

#+ : 1번이상 
bool(re.match('ca+t','cat'))
bool(re.match('ca+t','caat'))
bool(re.match('c+a','at time'))
bool(re.match('c+a','dat'))

#? : 0번 ,1번
bool(re.match('ca?t','cat'))
bool(re.match('ca?t','caat'))
bool(re.match('c?a','at time'))
bool(re.match('c?a','dat'))

#{n} : n번 반복
bool(re.match('ca{2}t','cat'))
bool(re.match('ca{3}t','cat'))
bool(re.match('ca{2}t','caat'))
bool(re.match('ca{2,3}t','caaaat'))
bool(re.match('c{2}a','at time'))
bool(re.match('c{2}a','dat'))

# | : or 
bool(re.match('c|a','at'))
bool(re.match('c|a','ct'))
bool(re.match('k|n|m','cm'))

#[] : 또는, 각각의 문자패펕
bool(re.match('[c,a]','at'))
bool(re.match('[c,a]','ct'))

bool(re.match('[abc]at','aat'))

bool(re.match('[ab]kkk','g'))

#[0-9] : 숫자
bool(re.match('[0-9]','3y'))
bool(re.match('[0-1]','3y'))
bool(re.match('[0-1]th','1th'))
bool(re.match('[0-9][0-9]','3y'))
bool(re.match('[0-9]*','3y'))
bool(re.match('[0-9]+','3y'))
bool(re.match('[0-9]?','3y'))
bool(re.match('[0-9]{2}','3y'))

re.match('[0-9]{3}\-[0-9]{4}\-[0-9]{4}','010-1111-1111')

#\d : 숫자와 매치 = [0-9]
bool(re.match('\d\dth','21th'))
bool(re.match('\d*th','21th'))
bool(re.match('\d+th','21th'))
bool(re.match('\d?th','21th'))
bool(re.match('\d{2}th','21th'))

#[a-z] : 알파뱃 
bool(re.match('[a-z]','cat'))
bool(re.match('[a-z]','Cat'))

#[A-Z] : 알파뱃
bool(re.match('[A-Z]','cat'))
bool(re.match('[A-Z]','Cat'))

#[a-zA-Z] 대소문자 알파뱃 모두
bool(re.match('[a-zA-Z]','cat'))
bool(re.match('[a-zA-Z]','Cat'))
bool(re.match('[a-zA-Z]','CAT'))

#[a-zA-Z0-9] 대소문자 알파뱃 숫자 모두
bool(re.match('[a-zA-Z0-9]','cat'))
bool(re.match('[a-zA-Z0-9]','21cat'))
bool(re.match('[a-zA-Z0-9]','Cat'))
bool(re.match('[a-zA-Z0-9]','CAT'))

#\w : 대소문자 알파뱃, 숫자 모두
bool(re.match('\w','21cat'))

#\W : 알파뱃, 숫자 아닌 문자검색
bool(re.match('\W','#'))
bool(re.match('[^a-zA-Z0-9]','#'))

#\D : 숫자 아닌 문자 검색
bool(re.match('\D','123'))
bool(re.match('\D','#'))
    
#\s : 공백 
bool(re.match('\sand','r and python'))#False

bool(re.search('\sand','r and python'))#True

#\S : 공백 문자 아닌 문자 
bool(re.search('\Sand','r and python'))

if re.match('Da','Data Science'):
    print('패턴이 일치')
else:
    print('패턴 불일치')
    
#re.I : 대소문자 무시
if re.match('da','Data Science', re.I):
    print('패턴 일치')
else:
    print('패턴 불일치')

m = re.match('Data','Data Science')
m.group()#'Data'
m.start()#0
m.end()#4
m.span()#(0,4)

m = re.search('Science','Data Science')
m.group()#'Science'
m.start()#5
m.end()#12
m.span()#(5,12)

m = re.search('a','Data Science')
m.group()#'a'
m.start()#1
m.end()#2
m.span()#(1,2)

#findall : 일치하는 문자열을 리스트로 반환
re.findall('a','Data Science')
re.findall('a.','Data Science')
re.findall('a.?','Data Science')
re.findall('a.*','Data Science')

#[문제 154] '오늘은 2019년 9월 30일입니다.' 이 문자열에서 숫자만 찾아주세요 
re.match('[0-9]','오늘은 2019년 9월 30일입니다.')
re.findall('[0-9]','오늘은 2019년 9월 30일입니다.')
#['2', '0', '1', '9', '9', '3', '0']
re.findall('[0-9]+','오늘은 2019년 9월 30일입니다.')
#['2019', '9', '30']
re.findall('\d+','오늘은 2019년 9월 30일입니다.')

#[문제 155] '오늘은 2019년 9월 30일입니다.' 이 문자열에서 문자만 찾아주세요 
re.findall('\D+','오늘은 2019년 9월 30일입니다.')
re.findall('[^0-9]+','오늘은 2019년 9월 30일입니다.')

re.findall('[a-zA-Z]+','오늘은 2019년 9월 30일입니다.')
re.findall('[가-힣]+','오늘은 2019년 9월 30일입니다.')
#숫자만 찾는다면 
re.findall('[^가-힣|.]+','오늘은 2019년 9월 30일입니다.')


##
source = 'Data Science'
#Science -> Scientist 으로 바꾸려 함 
source.replace('Science','Scientist')#'Data Scientist'
re.sub('Science','Scientist',source)#'Data Scientist'

#compile() 사용 
str = 'Physical Science and Computer Science and Logical Science'
#Physical -> Data
#Computer -> Data
p = re.compile('Physical|Computer')
#'Data Science and Data Science and Logical Science'
p.sub('Data',str)
#바뀐 갯수도 출력
p.subn('Data',str)
#('Data Science and Data Science and Logical Science', 2)

p = re.compile('\w+')
txt = "Let's live happily"
p.findall(txt)
#['Let', 's', 'live', 'happily']
q = re.compile('[a-z]+')
q.findall(txt)
#['et', 's', 'live', 'happily']

#[문제 156] '010101-1234567' -> ******-1234567 수정하세요
#단 replace를 이용하세요

x = '010101-1234567'
x.index('-')
x.replace(x[0:x.index('-')],'*'*6)#'******-1234567'

re.sub('\d{6}[-]\d{7}','******-1234567',x)
#() : group화, \g<> 안에 그룹번호 
re.sub('(\d{6})[-](\d{7})','******-\g<2>',x)#'******-1234567'
re.sub('(\d{6})[-](\d{7})','******-\g<1>',x)#'******-010101'

#[문제 157] 메일 주소를 검색해주세요
email_add = "test@naver.com \
            happy@gmail.com \
            dady@gmail.com \
            2019happy@daum.net \
            james@naver.com"
            
re.findall('\w+@\w+\.\w+',email_add)
re = re.compile('[\w]+@[\w]+[\.][\w]+')
re.findall(email_add)

#웹 크롤링을 해 보자 
#beautifulsoup
from bs4 import BeautifulSoup

#다음과 같은 html 파일이 있다고 하자 
html = """
<html>
<body>
<h1> 스크래핑 </h1>
<p> 웹페이지 분석하기 </p>
<p> 데이터 정제 작업하기 1 </p>
<p> 데이터 정제 작업하기 2 </p>
</body>
</html>"""
html

soup = BeautifulSoup(html, "html.parser")
soup
#<html>
#<body>
#<h1> 스크래핑 </h1>
#<p> 웹페이지 분석하기 </p>
#<p> 데이터 정제 작업하기 1 </p>
#<p> 데이터 정제 작업하기 2 </p>
#</body>
#</html>
#다음처럼 테그를 따라 갈 수 있다.
h1 = soup.html.body.h1
h1#<h1> 스크래핑 </h1>
h1.string#' 스크래핑 '

p = soup.html.body.p
p#<p> 웹페이지 분석하기 </p>
p.string#' 웹페이지 분석하기 '

#다음 p 테그에 있는거 보려면 
p.next_sibling#\n
p2 = p.next_sibling.next_sibling
p2#<p> 데이터 정제 작업하기 1 </p>
p2.string#' 데이터 정제 작업하기 1 '

html = """
<html>
<body>
<h1 id ='title'> beautifulsoup</h1>
<p id = 'subtitle'> 스크래핑</p>
<p> 데이터 추출하기 </p>
</body>
</html>"""
html

soup = BeautifulSoup(html, "html.parser")
soup.html.body.h1.string

soup.find(id = 'title')#<h1 id="title"> beautifulsoup</h1>
soup.find(id='title').string#' beautifulsoup'

html="""
<html>
<body>
<ul>
<li><a href = 'http://www.itwill.com'>아이티윌</a></li>
<li><a href = 'http://www.naver.com'>네이버</a></li>
</ul>
</body>
</html>"""
soup = BeautifulSoup(html, "html.parser")   
soup.html.body.ul.li.a.string#'아이티윌'

li1 = soup.html.body.ul.li
li1.string#'아이티윌'
li2 = li1.next_sibling.next_sibling
li2.a.string#'네이버'

li1.a.attrs['href']
li2.a.attrs['href']

link = soup.find_all('a')
link

for i in link:
    print(i.attrs['href'])
    print(i.string)
#http://www.itwill.com
#아이티윌
#http://www.naver.com
#네이버   
    

##
html = """
<html>
<body>
<h1> 스크래핑 </h1>
<p> 웹페이지 분석하기 </p>
<p> 데이터 정제 작업하기 1 </p>
<p> 데이터 정제 작업하기 2 </p>
</body>
</html>"""
html

soup = BeautifulSoup(html, "html.parser") 
soup

#p 테그 글을 뽑아서 보려면 
for i in soup.find_all('p'):
    print(i.string)

#파일을 받고 열어보자 (ctrl + s) 
with open('C:\\WorkSpace\\Python_Space\\data\\a.html') as html:
    soup = BeautifulSoup(html, "html.parser")
    print(soup)
    
    for i in soup.find_all('p'):
        print(i.string)
    #두번째 것이 출력이 안된다. <br> 때문이다. 다른 함수로 바꾸어 보자 
    for i in soup.find_all('p'):
        print(i.get_text())

soup.find('body')
soup.find('body').string#출력되는게 없다. 

#각 p 테그에 있는 글을 보려면 
for i in soup.find('body'):
    print(i.string)

#get_text() 
soup.find('body').get_text()
soup.find('body').get_text(strip = True)#\n이 없이 나온다.

#다음은 작동하지 않는다. error 발생 
for i in soup.find('body'):
    print(i.get_text())

#find_all 로 바꾸면 잘 작동한다. 
for i in soup.find_all('body'):
    print(i.get_text())
    
for i in soup.findAll('body'):
    print(i.get_text())
    
soup.find_all('a')[0].attrs['href']#'https://www.itwill.co.kr/'
soup.find_all('a')[1].attrs['href']#'http://www.naver.com/'
soup.find_all('a')[2].attrs['href']#'http://www.google.com/'

url = []
for i in soup.find_all('a'):
    print(i.attrs['href'])
    print(i.string)
    url.append(i.attrs['href'])
url

#a 테그 class="cafe1" 만 
soup.find('a',{'class':'cafe1'})
soup.find('a',{'class':'cafe1'}).attrs['href']

soup.find('a',{'class':'cafe2'})
soup.find('a',{'class':'cafe2'}).attrs['href']

soup.find(['a'])
soup.find(['p'])
soup.find(['a','p'])#a나 p중에서 먼저 줄에 있는게 하나 출력된다.
soup.find_all(['a','p'])#a나 p중에 해당되는게 모두 출력된다.
soup.findAll(['a','p'])
for i in soup.findAll(['a','p']):
    print(i.get_text())

#문자가 포함된 부분을 출력함 
#import re 했었는데 안 될때가 있다. 다시 import를 하자 
soup.find(text = re.compile('환영'))
soup.findAll(text = re.compile('이'))

soup.findAll('p')
soup.findAll('p',limit = 2)

#https://, http://, ftp://
soup.find_all('a',attrs = {'href':re.compile('https://')})

html="""
<html>
<body>
<div id = 'lecture1'>
<h1> 데이터 과학 </h1>
</div>
<div id = 'lecture2'>
<h1> 데이터 분석 </h1>
<ul class = 'subject'>
<li> SQL </li>
<li> R </li>
<li> Python </li>
</ul>
</div>
</body>
</html>"""

soup = BeautifulSoup(html, 'html.parser')
soup.html.div.hi.string
soup.find('h1').string
soup.find('h1').get_text()
soup.find('h1').text

soup.findAll('h1')

for i in soup.findAll('h1'):
    print(i.get_text())
    
    
h1 = soup.find(id='lecture1')
h1.string#
h1.get_text()#'\n 데이터 과학 \n'
h1.text#'\n 데이터 과학 \n'

h1.find('h1')#<h1> 데이터 과학 </h1>
h1.find('h1').string#' 데이터 과학 '

h1 = soup.find(id = 'lecture2')
h1
h1.find('h1')#<h1> 데이터 분석 </h1>
h1.find('h1').string#' 데이터 분석 '

ul = soup.find(id = 'lecture2')
ul = ul.find('ul')
ul

div = soup.find(id = 'lecture2')
ul = div.find('ul')
ul.findAll('li')#[<li> SQL </li>, <li> R </li>, <li> Python </li>]

soup.find('ul',{'class':'subject'})
soup.find(class_='subject')

#[문제158] 이주소로 접속하셔서 게시글을 출력하세요.
#http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?hmpMnuId=106
import urllib.request as req

url = "http://home.ebs.co.kr/ladybug/board/6/10059819/\
oneBoardList?c.page=1&hmpMnuId=106&searchKeywordValue=0\
&bbsId=10059819&searchKeyword=&searchCondition=&searchConditionValue=0&"
url
#html을 긁어온다 
res = req.urlopen(url)
res
soup = BeautifulSoup(res, "html.parser")
soup
#게시글이 있는곳을 지정한다.
p1 = soup.find_all('p',{'class':'con'})

for i in p1:
    print(i.string)
    
for i in p1:
    print(i.get_text())
    
data = []
for i in p1:
    data.append(i.get_text())
data

#[문제 159] 게시글 뿐만 아니라 게시날짜 정보도 같이 출력하세요
date = soup.find_all('span',{'class':'date'})

data2 = []
for i in date:
    data2.append(i.get_text())
data2

##둘다 한 번에 list에 저장하려면 
a = soup.find_all('p',class_='con')
b = soup.find_all('span',class_='date')
print(a[0].get_text())
print(b[0].text)

data = []
cnt = 0
for i in a:
    print(b[cnt].text, i.get_text(strip = True))
    data.append(b[cnt].text + '  ' + i.get_text(strip=True))
    cnt += 1
data

#이건 안되는 건가 
c = soup.find_all(('p',class_='con'),('span',class_='date'))
