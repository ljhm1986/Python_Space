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
#re library를 불러들인다. 
import re
#매타문자 
#. : 문자를 의미 
#

#re.match(찾을문자패턴, 문자열) : 오직 문자열의 시작점에서만 맞추어보기
#re.search(찾을문자패턴, 문자열) : 문자열안 어디에서나 맞추어보기 
#두 함수가 다르기 때문에 결과가 다르게 나오는게 많다. 
re.match('a','aa')
re.match('c.t','cat')
#<re.Match object; span=(0, 3), match='cat'>#문자열 0번부터 3번 이전까지 매칭
re.match('c.t','caat cat')
re.search('c.t','caat cat')

#. : 문자 한글자를 의미
bool(re.match('c.t','cat'))
bool(re.match('c.t','caat cat'))
re.match('c\.t','caat cat c.t')#. 이 있는경우 \.으로 찾기 
re.search('c\.t','caat cat c.t')
bool(re.match('c\.t','caatcatc.t'))#False
bool(re.search('c\.t','caatcatc.t'))#True
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
bool(re.match('k|n|m','cm km nm'))#False
bool(re.search('k|n|m','cm km nm'))#True

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
re.search('[0-9]{3}\-[0-9]{4}\-[0-9]{4}','010-1111-1111')

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
bool(re.search('\Sand','r and python'))#False
bool(re.search('\Sand','r aand python'))#True

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

#re.findall(찾는문자패턴, 문자열) : 일치하는 문자열들을 리스트로 반환
re.findall('a','Data Science')#['a', 'a']
re.findall('a.','Data Science')#['at', 'a ']
re.findall('a.?','Data Science')#['at', 'a ']
re.findall('a.*','Data Science')#['ata Science']

#[문제 154] '오늘은 2019년 9월 30일입니다.' 이 문자열에서 숫자만 찾아주세요 
re.match('[0-9]','오늘은 2019년 9월 30일입니다.')
re.search('[0-9]','오늘은 2019년 9월 30일입니다.')
re.findall('[0-9]','오늘은 2019년 9월 30일입니다.')
#['2', '0', '1', '9', '9', '3', '0']
re.findall('[0-9]+','오늘은 2019년 9월 30일입니다.')
#['2019', '9', '30']
re.findall('\d+','오늘은 2019년 9월 30일입니다.')
#['2019', '9', '30']

#[문제 155] '오늘은 2019년 9월 30일입니다.' 이 문자열에서 문자만 찾아주세요 
re.findall('\D+','오늘은 2019년 9월 30일입니다.')
re.findall('[^0-9]+','오늘은 2019년 9월 30일입니다.')

re.findall('[a-zA-Z]+','오늘은 2019년 9월 30일입니다.')
re.findall('[가-힣]+','오늘은 2019년 9월 30일입니다.')
#숫자만 찾는다면 
re.findall('[^가-힣|.]+','오늘은 2019년 9월 30일입니다.')


##문자열의 단어를 교체해보자 
source = 'Data Science'
#Science -> Scientist 으로 바꾸려 함 
source.replace('Science','Scientist')#'Data Scientist'
#위와 같은 기능을 가진 함수는 
#re.sub(찾는문자패턴, 바꾸려는문자, 문자열, count = 0)
re.sub('Science','Scientist',source)#'Data Scientist'

#re.compile(찾는문자패턴) 사용 
#아래처럼 다른 함수와 같이 사용된다.
#pattern = re.compile(찾는문자패턴)
#result = pattern.match(문자열)
str = 'Physical Science and Computer Science and Logical Science'
#Physical -> Data
#Computer -> Data
p = re.compile('Physical|Computer')
p.match('Physical')
p.sub('Data',str)
#'Data Science and Data Science and Logical Science'
#re.subn() : re.sub()과 같은 기능, 바뀐 갯수도 출력
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
#re로 저장하면 import re 다시해야한다.
re1 = re.compile('[\w]+@[\w]+[\.][\w]+')
re1.findall(email_add)

re2 = re.compile(r"[a-zA-Z0-9]+@[a-zA-Z0-9]+[\.][a-zA-Z0-9]+")
re2.findall(email_add)

############################
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

#html file을 열기 
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

#soup.find(name, attrs, recursive, string, **kwargs)
soup.find(id = 'title')#<h1 id="title"> beautifulsoup</h1>
type(soup.find(id = 'title'))#bs4.element.Tag
#.string : 해당값에 대한 태그가 하나만 있고 type이 NavigableString일때 가능 
soup.find(id='title').string#' beautifulsoup'

#soup.find_all(name, attrs, recursive, string, limit, **kwargs)
soup.find_all(id='title')#[<h1 id="title"> beautifulsoup</h1>]
type(soup.find_all(id='title'))#bs4.element.ResultSet
#.string은 ResultSet일때 되지 않는다 
soup.find_all(id='title').string

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
#.next_sibling : 같은 레벨의 태그중 뒤에 것 
#.previous_sibling : 같은 레벨의 태그중 앞의 것
li2 = li1.next_sibling.next_sibling
li2.a.string#'네이버'

li1.a.attrs['href']
li2.a.attrs['href']

link = soup.find_all('a')
link
type(link)#bs4.element.ResultSet
link[0]

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
type(soup.find('body'))
soup.find('body').string#출력되는게 없다. 
type(soup.find('body').string)

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

#a 테그 class="cafe1" 만 골라내기 
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

###################################################################
#10/1#
#중앙일보 '인공지능' 검색
import re
from bs4 import BeautifulSoup
import urllib.request as req

url = "https://search.joins.com/JoongangNews?page=1\
&Keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%20\
&SortType=New&SearchCategoryType=JoongangNews"

html = req.urlopen(url)
html
soup = BeautifulSoup(html, "html.parser")
soup

txt = soup.find_all('h2',{'class':'headline mg'})
txt
len(txt)

#이제 기사링크들을 list에 저장하자 
url_col = []
for i in txt:
    print(i.get_text())
    print(i.find('a').attrs['href'])
    url_col.append(i.find('a').attrs['href'])
url_col

#기사 제목    
txt[0].string
#기사 링크 
txt[0].find('a').attrs['href']

#다르게도 표현 
for i in soup.findAll('h2',{'class':'headline mg'}):
    print(i.get_text())
    print(i.string)
    print(i.find('a').attrs['href'])
    print(i.find('a').get('href'))
    
#모든(?) 링크들을 나타낸다.
soup.find_all('a',attrs = {'href':re.compile('https://')})

#특정 기사의 본문을 스크랩 하자
url = "https://news.joins.com/article/23591525"

html = req.urlopen(url)
html
soup = BeautifulSoup(html, "html.parser")
soup
type(soup)

txt = soup.find('div',id='article_body').get_text(strip = True)
txt


##이제 여러 페이지에 나오는 것을 기사 링크를 수집해보자 
url = "https://search.joins.com/JoongangNews?page={}\
&Keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%20\
&SortType=New&SearchCategoryType=JoongangNews"

url.format(1)

soups = []
url_col = []
for i in range(10):
    html = req.urlopen(url.format(i))
    soup = BeautifulSoup(html, "html.parser")
    soups.append(soup)
    txt = soup.find_all('h2',{'class':'headline mg'})
    for j in range(len(txt)):
        url_col.append(txt[j].find('a').attrs['href'])

soups
type(soups)
url_col

#선생님 풀이#
url ="https://search.joins.com/TotalNews?Keyword=\
%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&SortType=New\
&SearchCategoryType=TotalNews&PeriodType=All\
&ScopeType=All&ImageType=All&JplusType=All\
&BlogType=All&ImageSearchType=Image&TotalCount=0\
&StartCount=0&IsChosung=False&IssueCategoryType=\
All&IsDuplicate=True&Page=1&PageSize=10&IsNeedTotalCount=True"

html = req.urlopen(url)
soup= BeautifulSoup(html, "html.parser")
h2 = soup.find_all('h2',class_="headline mg")

url_2 = []

for i in h2:
    print(i.find('a').attrs['href'])
    url_2.append(i.find('a').attrs['href'])

url_2 = []
for i in h2:
    print(i.find('a').get('href'))
    url_2.append(i.find('a').get('href'))


news = []
for i in url_2:
    html = req.urlopen(i)
    soup= BeautifulSoup(html, "html.parser")
    news.append(soup.find('div',id="article_body").get_text(strip=True))

news[0]

###############
url_2 = []

url = "https://search.joins.com/JoongangNews?page={}\
&Keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&SortType=New\
&SearchCategoryType=JoongangNews"

#기사 링크들을 모으자 
for i in range(1,11):    
    html = req.urlopen(url.format(i))
    soup= BeautifulSoup(html, "html.parser")
    h2 = soup.find_all('h2',class_="headline mg")

    for i in h2:
        print(i.find('a').attrs['href'])
        url_2.append(i.find('a').attrs['href'])

#링크된 주소의 기사 본문을 모으자 
news = []
for i in url_2:
    html = req.urlopen(i)
    soup= BeautifulSoup(html, "html.parser")
    news.append(soup.find('div',id="article_body").get_text(strip=True))

news[0]

##이번에는 동아일보 기사를 수집해보자 ##
url = "http://www.donga.com/news/search?p=1\
&query=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5\
&check_news=1&more=1&sorting=1&search_date=1&v1=&v2=&range=1"
list(range(1,200,15))

html = req.urlopen(url)
soup = BeautifulSoup(html, "html.parser")
#다음과 같은 문구가 뜨면 나중에 다시 해 보자 
#ConnectionResetError: [WinError 10054] 
#현재 연결은 원격 호스트에 의해 강제로 끊겼습니다
#계속 안 되는데... 

news_link = []
for i in soup.find_all('p',class_= 'tit'):
    print(i.find('a').attrs['href'])
    news_link.append(i.find('a').attrs['href'])
    
news_link#뉴스들의 링크가 저장된걸 확인할 수 있다 

news_txt = []
for i in news_link:
    html = req.urlopen(i)
    soup = BeautifulSoup(html, "html.parser")
    news_txt.append(soup.find('div',class_='article_txt').get_text(strip = True))

news_txt

#선생님의 풀이#
import urllib.request as req
from bs4 import BeautifulSoup

params = []

url = "http://news.donga.com/search?p={}\
&query=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&\
check_news=1&more=1&sorting=1&search_date=1&v1=&v2=&range=1"

for i in range(1,137,15): 
    res = req.urlopen(url.format(i))
    soup= BeautifulSoup(res, "html.parser")
    for link in soup.findAll('p', {'class':'tit'}):
        params.append(link.find('a').get('href'))

params
txt= []
error = []

for i in params:
    try:
        #print(i)
        res = req.urlopen(i)
        soup= BeautifulSoup(res, "html.parser")
        result = soup.find_all('div',class_='article_txt')
        for i in result:
            #print(i.text)
            txt.append(i.text)
    except:
        error.append(i)
    

txt[0]
txt[0][0:txt[0].find('Copyright')]

new_txt = []
for i in range(0,len(txt)):
    new_txt.append(txt[i][0:txt[i].find('Copyright')])
new_txt


##
html = """
<ul id = "조선왕">
<li id = "태조"> 이성계 </li>
<li id = "정종"> 이방과 </li>
<li id = "태종"> 이방원</li>
<li id = "세종"> 이도</li>
<li id = "문종"> 이향</li>
</ui>"""

soup = BeautifulSoup(html, "html.parser")
soup.find('li',id='세종').get_text()
soup.select_one('li#세종')
type(soup.select_one('li#세종'))#bs4.element.Tag
soup.select_one('li#세종').text
soup.select('li#세종')
type(soup.select('li#세종'))#list 
type(soup.select('li#세종')[0])#bs4.element.Tag          
        
soup.select_one('li:nth-of-type(1)').text
soup.select_one('li:nth-of-type(1)').text.strip()

soup.select_one('li') #find
soup.select('li') #find_all, findAll

for i in soup.select('li'):
    print(i.string)
    
soup.select('li')[0].string

#네이버 금융에서 일본 앤화와의 환율을 스크래핑
url = "https://finance.naver.com/marketindex/"
url
html = req.urlopen(url)
soup = BeautifulSoup(html, "html.parser")
fin = soup.find_all('div',class_='head_info point_dn')
fin[0].find('span',class_='value').get_text()

#선생님의 풀이 #
soup.select_one('ul#exchangeList')
# #는 id를 의미한다. 바로 자식테그 li 중에 2번째로 간다. 
a = soup.select_one("ul#exchangeList > li:nth-of-type(2)")
a.select("span:nth-of-type(1)")#이러면 테그 경로로 첫번째 span이 다 나온다.
a.select("span:nth-of-type(1)")[1].text
a.select_one("span:nth-of-type(2)")

# 이번에는 div를 써서 바로 div테그에 속하는것만 나오게 했다.
a2 = soup.select_one("ul#exchangeList > li:nth-of-type(2) div")
a2.select_one("span:nth-of-type(1)").string
#'1,107.60'

#또는 다음과 같이 
a3 = soup.select_one("ul#exchangeList > li:nth-of-type(2) >\
                     a:nth-of-type(1) > div")
a3.select_one("span:nth-of-type(1)").string

#이미지 추출해서 저장하려면 
a4 = soup.select_one("ul#exchangeList > li:nth-of-type(2) >\
                     a:nth-of-type(2)")
lnk = a4.select_one("img").get("src")

import urllib.request as req
req.urlretrieve(lnk,"C:\\WorkSpace\\Python_Space\\data\\20191001.jpg")
#해당 폴더로 가서 보면 파일이 저장되어 있는걸 확인할 수 있다. 

#다움 뉴스
url = "https://news.v.daum.net/v/20191001121023244"
html = req.urlopen(url)
soup = BeautifulSoup(html, "html.parser")
soup
type(soup)
len(soup)

#뉴스 본문이 p테그 안 여러곳에 나누어져 있다. 
for i in soup.select("div#harmonyContainer p"):
    print(i.string)

news_article = []
for i in soup.select("div#harmonyContainer p"):
    news_article.append(i.get_text())

news_article
len(news_article)

#########################################################################
#10/2#
#네이버금융 - 국내증시 - 검색상위종목
#검색상위종목 회사들의 데이터들이 테이블 표 처럼 되어있다.
#<table> 테그 안에 데이터들이 들어가 있다. 
import urllib.request as req
from bs4 import BeautifulSoup

url = "https://finance.naver.com/sise/lastsearch2.nhn"
html = req.urlopen(url)
soup = BeautifulSoup(html, "html.parser")
type(soup)

table = soup.select('table.type_5')
type(table)
table[0].select('tr.type1 > th')

for i in table[0].select('tr.type1 > th'):
    print(i.string)

#DataFrame을 만들기 위해서 columns들을 먼저 만들자 
column_name = []
for i in table[0].select('tr.type1 > th'):
    column_name.append(i.string)

column_name

#데이터들의 특성을 파악하자, 공백줄을 어떻게 넘어갈 것인가 
table[0].select('tr')
len(table[0].select('tr'))
table[0].select('tr')[1]
table[0].select('tr')[2].select('td.no')
table[0].select('tr')[3]
table[0].select('tr')[4]
table[0].select('tr')[5]
table[0].select('tr')[6]
table[0].select('tr')[7]
table[0].select('tr')[8]

#<td class="no"> 테그가 있을때에만 데이터가 있다.
for i in range(len(table[0].select('tr'))):
    print(table[0].select('tr')[i].select_one('td.no'))

#데이터를 추출하기 
for i in table[0].select('tr')[2].select('td'):    
    #<span> 안에 있을경우에는 
    if i.string is None:
        print(i.select('span')[0].string.strip())
    #<td>안에 있을 경우에는 
    else:
        print(i.string)

#위의 2가지를 합쳐서 list 에 데이터를 저장하도록 하자 
data_list = []#데이터를 저장할 중첩list
for i in range(len(table[0].select('tr'))):
    #임시 저장소 
    temp_list = []
    #<td class="no"> 테그가 있을때에만 보자 
    if table[0].select('tr')[i].select_one('td.no') is not None:
        #데이터를 추출하고 list에 저장하기 
        for j in table[0].select('tr')[i].select('td'):
            #<span> 안에 있을경우에는
            if j.string is None:
                temp_list.append(j.select('span')[0].string.strip())
            #<td>안에 있을 경우에는
            else:
                temp_list.append(j.string)
    #<td class="no"> 테그가 없으면 통과 
    else:
        continue
    data_list.append(temp_list)

data_list

#이제 중첩list를 DataFrame에 저장하자 
from pandas import Series, DataFrame
import pandas as pd

fin_df = DataFrame(data_list, columns = column_name)

fin_df

### 선생님의 풀이
tr = soup.select('table.type_5 >tr')

for i in tr:
    print(i)
    
tr[0] #column name
type(tr[0])
tr[1] #no data
tr[2].select('td')
tr[6]
tr[7] #no data

#no data인 부분을 피하려면 어떻게 해야 할까?

tr[1].select('td.number')
tr[2].select('td.no')
tr[2].select('td > a')
tr[2].select('td.number')
tr[2].select('td.number')
tr[2].select('td.number > span')

for i in tr[2].select('td'):
    print(i.string)

tr[1].select_one('td.no')
type(tr[1].select_one('td.no'))#NoneType

#그러면 None인 부분을 건너뛰면 되겠다.
for i in tr:
    if i.select_one('td.no') is None:
        pass
    else:
        print(i.select_one('td.no'))
        
#이제 span 테그에 데이터가 있는것은 어떻게 처리하나 
cols = tr[2].select('td')
cols[0].select_one('span')
[i.select_one('span').string.strip() 
if str(i).find('span') != -1 
 else i.text.strip() for i in cols]

#위 2개를 합치자 
lst = []
for i in tr:
    if i.select_one('td.no') is None:
        pass
    else:
        cols = i.select('td')
        data = [x.select_one('span').string.strip() 
                if str(x).find('span') != -1 
                else x.text.strip() for x in cols]
        lst.append(data)

lst

col = []
for i in tr[0].select('th'):
    col.append(i.string)
    
col

df_data = DataFrame(lst, columns = col)
df_data

#한국농촌경제연구원 - 해외곡물정보 
#F12누르고 network로 들어가서 날짜 파일 누르면 url이 나온다.(F5 눌러서 새로고침)
#http://www.krei.re.kr:18181/chart/main_chart/index
#/kind/S/sdate/2010-01-01/edate/2019-10-02
#url을 검색하면 JSON형태의 데이터 파일이 나온다.
#JSON - 텍스트 데이터를 기반, 자바스크립트에서 사용하는 객체 표기 방법을 기반으로 한다.
#- 자바스크립트 전용데이터 형식은 아니고 다양한 소프트웨어와 
#프로그램 언어끼리 데이터 교환할때 사용 
import json
url = "http://www.krei.re.kr:18181/chart/main_chart/index\
/kind/S/sdate/2010-01-01/edate/2019-10-02"

import urllib.request as req
res = req.urlopen(url)
res

#json 형태로 로드하기 
json_res = json.load(res)
json_res
type(json_res)#list, list안에 dictionary형태로 들어있다.

json_res[0]

for i in json_res:
    print(i['date'],i['settlement'])
    
res.read()#한번 가져오면 없어지나..?
res = req.urlopen(url)
res_read = res.read()
res_read

#json 데이터를 pandas로 로드하자 
import pandas as pd
df = pd.read_json(res_read)
df
df.info()#DataFrame
df['date']

#이제 날짜와 가격만 뽑아서 그래프를 그려보자 
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)
pd.to_datetime(df['date'].astype('str'))
df['settlement']

#날짜와 가격만 있는 DataFrame을 만들자 
df2 = DataFrame({'date':pd.to_datetime(df['date'].astype('str')),
                 'settlement':df['settlement']})
df2
#plt.bar(df2['date'],df2['settlement']) 이건 피하도록 하자, 컴퓨터가 힘들어 한다 
plt.figure(figsize = (12,6))
plt.plot(df2['date'],df2['settlement'])
plt.title('세계 곡물 가격')
plt.xlabel('날짜')
plt.ylabel('가격')
plt.grid(True)

df2['settlement'].plot(grid = True, figsize = (12,6))
#그런데 이렇게만 하면 x축이 index가 나온다. 

#다음 - 증권 - 인기검색 10
#http://finance.daum.net/api/search/ranks?limit=10 링크 들어가면 
#403 Forbidden 이 뜬다. 봇인걸 막아놓는 경우가 있는데 이럴때는 
#Header를 만들어서 링크와 같이 가야한다.
#network에서 밑에 보면 Request Header 부분이 있는데 그 중에서 
#Referer: http://finance.daum.net/
#User-Agent: Mozilla/5.0 (Windows NT 6.1; Win64; x64) \
#AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) \
           AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 \
           Safari/537.36",
           "Referer": "http://finance.daum.net/"
           }
url = "http://finance.daum.net/api/search/ranks?limit=10"

req_data = req.urlopen(req.Request(url, headers = headers)).read().decode('utf-8')
req_data

json_data = json.loads(req_data)['data']
json_data

for i in json_data:
    print(i['rank'],i['name'],i['tradePrice'])

#업종 상위 - 코스피 를 뽑아보자 ... ?
#이것도 403 Forbidden 이 뜬다.
kospi_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64)\
                 AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90\
                 Safari/537.36",
                 "Referer": "http://finance.daum.net/"
                 }

kospi_url = "http://finance.daum.net/api/sectors?market=KOSPI&\
change=RISE&includedStockLimit=1&perPage=5"

kospi_data = req.urlopen(req.Request(kospi_url,headers = kospi_headers)
).read().decode('utf-8')

json_kospi_data = json.loads(kospi_data)['data']
json_kospi_data

for i in json_kospi_data:
    print(i['sectorName'],i['includedStocks'][0]['name'])
    
#################################################################
#10/4#
#selenium 
#웹브라우저를 컨트롤하여 웹 UI(User Interface)를 자동화하는 도구
#command line에서 selenium library 설치함

from selenium import webdriver
url = "http://naver.com"
driver = webdriver.Chrome("C:\WorkSpace//chromedriver.exe")
driver.get(url)
driver.save_screenshot("C:\WorkSpace\\Python_Space\\data\\naver.png")
driver.quit()

url = "http://daum.net"
driver = webdriver.PhantomJS("C:\WorkSpace//phantomjs.exe")
driver.get(url)#Selenium은 PhantomJS 지원을 deprecate 할꺼라..
#팝업창이 뜨지 않는다.
driver.save_screenshot("C:\WorkSpace\\Python_Space\\data\\daum.png")
#스샷은 저장된다. 어디까지 작동이 되나 디버깅용으로 사용된다고 하심
driver.quit()

#네이버 이미지 검색 
url = "https://search.naver.com/search.naver?where=image"
import urllib.request as req
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Chrome("C:\WorkSpace//chromedriver.exe")
driver.get(url)
ser = driver.find_element_by_id("nx_query")#검색어 입력 
ser.send_keys("다크소울 2")#검색어 
ser.submit()

#사진들이 나온걸 보면 모든 사진들이 나온게 아니고, 스크롤바를 내리면
#새로운 사진들이 로딩된다. 스크롤바를 내려보자 
for i in range(1,3):
    driver.find_element_by_tag_name('body').send_keys(Keys.END)
    time.sleep(5)#5초 대기

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
soup

params = []
img_list = soup.find_all("img", class_="_img")
img_list[99]
img_list[100]
img_list[102]
for i in img_list:
    params.append(i['src'])

params[100]

x = 1
for p in params:
    #링크주소가 있는 사진들을 해당 경로에 저장하자 
    req.urlretrieve(p,"C:\WorkSpace\\sample_img_data\\darksoul2_"+str(x)+".jpg")
    x += 1

params2 = []
img_list2 = soup.select('img._img')

img_list2
for i in range(len(img_list2)):
    img_list2[i] = img_list2[i].attrs['src']


driver.quit()

#다음 이미지 검색 
url = "https://search.daum.net/search?w=img&nil_search=btn&DA=NTB&enc=utf8&q="
driver = webdriver.Chrome("C:\WorkSpace//chromedriver.exe")
driver.get(url)
ser = driver.find_element_by_id("q")
ser.send_keys("위쳐 3")
ser.submit()

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
soup

params = []
img_list = soup.find_all("img", class_="thumb_img")
img_list[1]['src']
for i in img_list:
    params.append(i['src'])

params

x = 1
for p in params:
    req.urlretrieve(p,"C:\WorkSpace\\sample_img_data\\witcher3_"+str(x)+".jpg")
    x += 1

driver.close()   
    
##yes24 로그인 하기(yes24에 회원가입이 되어 있어야 한다.)
url = "https://www.yes24.com/Templates/FTLogin.aspx"
user = ""
mypas = ""

driver = webdriver.Chrome("C:\WorkSpace//chromedriver.exe")
driver.get(url)
#아이디 입력
inputid = driver.find_element_by_id("SMemberID")
inputid.clear()#기존값 삭제 
inputid.send_keys(user)
driver.implicitly_wait(3)
#비밀번호 입력
inputpas = driver.find_element_by_id("SMemberPassword")
inputpas.clear()#기존값 삭제 
inputpas.send_keys(mypas)
driver.implicitly_wait(3)
#가동 
loginbn = driver.find_element_by_css_selector("button#btnLogin")
loginbn.submit() 


driver.close()  
#yes24 잘 안되었음...                                            
                 
#그외 본인이 회원가입한 사이트로 
#daum로그인 해 보자 
url = "https://logins.daum.net/accounts/signinform.do?url=https%3A%2F%2Fwww.daum.net%2F"

user = ""
mypas = ""

driver = webdriver.Chrome("C:\WorkSpace//chromedriver.exe")
driver.get(url)
#아이디 입력
inputid = driver.find_element_by_id("id")
inputid.clear()#기존값 삭제 
inputid.send_keys(user)
driver.implicitly_wait(3)
#비밀번호 입력
inputpas = driver.find_element_by_id("inputPwd")
inputpas.clear()#기존값 삭제 
inputpas.send_keys(mypas)
driver.implicitly_wait(3)
#가동 
loginbn = driver.find_element_by_css_selector("button#loginBtn")
loginbn.submit() 

driver.close()

####다시 네이버 로그인 
url = 'https://nid.naver.com/nidlogin.login'
driver = webdriver.Chrome("C:\WorkSpace//chromedriver.exe")
driver.implicitly_wait(5)
driver.get(url)
userid = 'ljhm_1986'
userpw = ''

driver.execute_script("document.getElementsByName('id')[0].\
                      value=\'"+userid+"\'")
driver.execute_script("document.getElementsByName('pw')[0].\
                      value=\'"+userpw+"\'")
driver.implicitly_wait(3)

driver.find_element_by_xpath('//*[@id="frmNIDLogin"]/fieldset/input').click()
driver.implicitly_wait(3)
#브라우저 등록 물어보면 (등록 안 하는거로...)
driver.find_element_by_xpath('//*[@id="frmNIDLogin"]\
                             /fieldset/span[2]/a').click()

driver.close()

####다나와
#XPATH들을 보면 
#열기
'//*[@id="dlMaker_simple"]/dd/div[2]/button[1]'
#닫기
'//*[@id="dlMaker_simple"]/dd/div[2]/button[2]'
#apple 클릭
'//*[@id="selectMaker_simple_priceCompare_A"]/li[14]'
'//*[@id="searchMaker1452"]'

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

url = 'http://prod.danawa.com/list/?cate=112758'
driver = webdriver.Chrome("C:\WorkSpace//chromedriver.exe")
driver.implicitly_wait(5)
driver.get(url)
#페이지가 뜰떄까지 기라리자 
#클릭 , 노트북 제조회사 목록을 더 보기 
WebDriverWait(driver, 3).until(EC.presence_of_element_located((
                By.XPATH,
                '//*[@id="dlMaker_simple"]/dd/div[2]/button[1]'))).click()

driver.implicitly_wait(3)

WebDriverWait(driver, 3).until(EC.presence_of_element_located((
                By.XPATH,
                '//*[@id="dlMaker_simple"]/dd/div[2]/button[2]'))).click()

driver.implicitly_wait(3)

#그리고 노트북 목록 중에서 하나 눌러보기 
WebDriverWait(driver, 3).until(EC.presence_of_element_located((
                By.XPATH,
                '//*[@id="adReaderProductItem9261852"]/div/div[2]/p/a'))).click()
#새로운 팝업창이 뜬다. 
driver.implicitly_wait(3)

#노트북 목록에서 이미지와 가격을 추출해보자 
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
soup
list_all = soup.select('div.main_prodlist.main_prodlist_list > ul > \
                       li.prod_item.prod_layer')
x = 1
for i in list_all:
    #인기순위와 제품명
    print(i.select_one("p.prod_name").text.strip())
    #이미지 링크
    print(i.select_one("a.thumb_link > img")['data-original'])
    #이미지 추출 
    req.urlretrieve(i.select_one("a.thumb_link > img")['data-original'],
                    "C:\WorkSpace\\sample_img_data\\notebook_"+str(x)+".jpg")
    x += 1
    #가격 
    print(i.select_one("p.price_sect > a > strong").text.strip())
    
#list_all2 = soup.find_all('div',class_='main_prodlist main_prodlist_list')
#list_all2[1]

#페이지를 넘기기 
num = 2
WebDriverWait(driver, 3).until(EC.presence_of_element_located((
        #XPATH 일때는 숫자를 넣어주어야 한다. css로 해서 nth-child()를 이용하자
        #안에 숫자만 바꾸어 넣으면 다른 목록 페이지로 넘어갈 수 있다.  
                By.CSS_SELECTOR,'div.number_wrap > a:nth-child({})'.format(num)
                ))).click()
#해 보면 제 2페이지로 넘어간걸 볼 수 있다. 
driver.close()


######페이지들 수집하기 ######
url = 'http://prod.danawa.com/list/?cate=112758'
driver = webdriver.Chrome("C:\WorkSpace//chromedriver.exe")
driver.implicitly_wait(5)
driver.get(url)

#apple사 제품만 나오도록 하자 
WebDriverWait(driver, 3).until(EC.presence_of_element_located((
                By.XPATH,'//*[@id="searchMakerRep1452"]'))).click()

time.sleep(5)#5초 기다렸다가 하자 
#바로 하면 페이지가 바뀌기 전에 클릭을 하게 되어서 error가 난다. 

for i in range(2,4):
    list_all3 = []
    #해당 페이지로 이동
    WebDriverWait(driver, 3).until(EC.presence_of_element_located(( 
                By.CSS_SELECTOR,'div.number_wrap > a:nth-child({})'.format(i)
                ))).click()
    
    #노트북 목록에서 이미지와 가격을 추출해보자 
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    list_all3 = soup.select('div.main_prodlist.main_prodlist_list > ul > \
                            li.prod_item.prod_layer')
    x = 1
    for j in list_all3:
        #인기순위와 제품명
        print(j.select_one("p.prod_name").text.strip())
        #이미지 링크
        print(j.select_one("a.thumb_link > img")['data-original'])
        #이미지 추출 
        req.urlretrieve(j.select_one("a.thumb_link > img")['data-original'],
                    "C:\WorkSpace\\sample_img_data\\notebook_"+
                    str(i)+'_'+str(x)+".jpg")
        x += 1
        #가격 
        print(j.select_one("p.price_sect > a > strong").text.strip())

#HTTPError: Forbidden 이거 뜨면 ...
driver.close()
#######