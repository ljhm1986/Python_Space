# -*- coding: utf-8 -*-
#위는 utf-8 로 인코딩 하겠다는 의미이다.
"""
Created on Mon Sep  2 10:46:11 2019

@author: stu4
"""
############################################
#9/2#
import math

math.pow(2,3)
2**3#8 #R : 2^3, 2**3

#변수(variable)
#데이터를 저장할 수 있는 메모리 공간
#첫글자는 영문, _(밑줄)
#두번째 글자부터는 영문자, 숫자, _ 
#대소문자 구분한다.
#예약어는 사용할 수 없다.
import keyword
keyword.kwlist

x = 1
x
print(x)
type(x)#int #R : mode(), class(), str()
dir()#현재 메모리에 있는 변수정보와 라이브러들을 확인할 수 있다.
del(x)#메모리에 있는 변수를 삭제하기 #R : rm()

locals()#x = 1 인걸 확인할 수 있다. dir()보다 더 상세하게 나온다.
#메모리에 있는 변수의 값을 확인

#연산자 축약으로 사용
x = 1
x = x + 1
x += 1
x

y = x - 1
y -= 1
y

z = x * 1
z *= 2

z = z/2
z /= 2

x = x //2 #나누기의 몫 #R : %%
x //=2
x

f = 10.12#실수형, 부동소수점(float)
type(f)#float

f = 10.4e3 # 10.4 * 10 ** 3 #R : 10.4e3
f
type(f)#float

#논리연산자 #대소문자 구분 주의
x = 1
y = 2
y > x
y >= x
y < x
y <= x
y == x
y != x

2 > 1 and 3 > 2
2 > 1 or 4 < 2
not 1 > 2

'''
[문제1] x,y 변수에 있는 값을 기준으로 수행한 결과 입니다. 
x 와 y 변수에 어떤 값이 있어야 하나요.
또한 결과값이 나오기 위해서 어떤 계산식을 만들어야 하는지 계산식을 만들어 보세요.


result_1 =  7
result_2 =  3
result_3 =  -3
result_4 =  10
result_5 =  0.4
result_6 =  0
result_7 =  2
result_8 =  32
result_9 =  7.0
result_10 =  -21
result_11 =  50
result_12 =  29
'''
x = 2
y = 5

result_1 = x + y
print(result_1)
result_2 = y - x
print(result_2)
result_3 = x - y
print(result_3)
result_4 = x * y
print(result_4)
result_5 = x / y
print(result_5)
result_6 = x // y
print(result_6)
result_7 = x % y
print(result_7)
result_8 = x ** y
print(result_8)
result_9 = (x+y)/x*x
print(result_9)
result_10 = (x - y) * (x + y)
print(result_10)
result_11 = x * y ** x
print(result_11)
result_12 = x ** x + y ** x
print(result_12)

print("result_1 = ", x +y)
print('result_1 = ', x +y)

#문자열
'대한민국'
"짝짝짝"

"""대한민국
짝짝짝"""
'''한조
대기 중'''
# \n 줄바꿈의 역활
'겐지가\n함께한다'
print('내가\n용이 된다')
#\t : tab키
print('오늘 하루도\t행복하자')
# \0 : null
print('python\0python')

#문자열 안에 문자열 표현
print('python \'python\'')#python 'python'
print("python 'python'")#python 'python'
#문자열 안에 \ 표현 
print("python \\python")#python \python

x = '이재훈'
y = '파이썬 개발자'
type(x)#str
x + y #문자 + 문자 = 문자 (연결연산자)

(x + y) * 2

print("="*50)
print("hello world")
print("="*50)

#문자열 안에 변수 문자열을 넣기 
name = 'lee jae hoon'
music = 'classic'
print('hello, {}입니다. \n 즐겨듣는 음악은 {} 입니다.'.format(name,music))
#hello, lee jae hoon입니다. 
#즐겨듣는 음악은 classic 입니다.
print('hello, %s입니다. \n 즐겨듣는 음악은 %s 입니다.'%(name,music))

x = 996
y = 8
result = x % y
print("{}를 {}나누면 {}가 나머지입니다.".format(x,y,result))
#996를 8나누면 4가 나머지입니다.
print("%d를 %d나누면 %d가 나머지입니다."%(x,y,result))

result1 = x //y
result2 = x % y
print("%d를 %d나누면 %d가 몫이고 %d는 나머지입니다."
      %(x,y,result1,result2))

divmod(x,y)#몫과 나머지가 나온다.
result1, result2 = divmod(x,y)
print("%d를 %d나누면 %d가 몫이고 %d는 나머지입니다."
      %(x,y,result1,result2))

print("원주율은 {}입니다.".format(3.14159))
print("원주율은 %d입니다."%(3.14159))#원주율은 3입니다.
print("원주율은 %f입니다."%(3.14159))#원주율은 3.141590입니다.

x = '라인하르트 여기 대령 했소이다'
y = '(겐지) 아아 여기가 나의 고향이다'
z = '(겐지하르트) 돌진!'
type(x)#str
#문자열의 문자 갯수 #R : nchar(), stringr :: str_length()
len(x)#17

#인덱싱, 0부터 #R : 1부터 
x[0]
x[-1]

#슬라이싱 #R : [시작요소번호:종료요소번]
x[:]
x[0:5] #[시작요소번호:큰요소번호이전까]
x[:5]
x[5:]
x[4:6]
x[4:-5]
x[::1]#하나씩 증가하며 뽑아냄
x[::2]#둘씩 증가하며 뽑아냄
x[1:7:2]#[시작점:끝점-1:증가분]
y[1:7:2]
z[1:7:2]
x[5::2]
#역순으로 보기 
x[::-1]

'''
[문제_2] v_str 변수에 이 문자열을 입력하세요.
v_str = "시간은 멈추지 않습니다. 하루를 유익하게 살아야합니다."

1. "시간은 멈추지 않습니다." 만 출력해주세요
2. "하루를 유익하게 살아야합니다." 만 출력해주세
3. "살아야합니다."  만 출력해주세요
4. "시추니루하야"  이 글자만 출력해주세
5. "시간은 멈추지 않습니다. 하루를 유익하게" 만 출력해주세요.
6 v_str 문자열을 뒤순으로 출력해 주세요.
'''
v_str = "시간은 멈추지 않습니다. 하루를 유익하게 살아야합니다."
v_str[0:12]
v_str[12:]
v_str[-7:]
v_str[0::5]
v_str[0:-7]
v_str[::-1]

#[문제3] 
x = '파리썬'
#인덱스를 이용해서 리 -> 이로 변환하세요.

#R : x[1] <- '이' 였음
x[1] = '이' #수정이 안 된다.
x = x[0]+'이'+x[2]
x

#문자함수
x

#원본문자를 찾아서 다른 문자로 치환하는 함수 
# R : stringr :: str_replace(x, '리','이')
x = x.replace('리','이')

x = 'hello world'
#원본 문자열이 매개변수로 입력한 문자열로 시작되는지
#판단하는 함수
x.startswith('h')
x.startswith('H')

#원본 문자열이 매개변수로 입력한 문자열로 끝나는지
#판단하는 함수
x.endswith('ld')
x[-2:] == 'ld'

x.endswith('D')
x[-1] == 'D'
x[len(x)-1] == 'D'

#원본문자열안에서 매개변수로 입력한 문자열이
#존재하는 위치를 앞에서부터 찾는다. 만약에 
#존재하지 않으면 -1로 나온다.
#R : stringr :: str_replace(x,'w')
x.find('w')#w가 처음 나오는 위치
x.find('world')#world가 처음 나오는 위치
x.find('W')#-1

cn = x.find('l')
cn

cn = x.find('l',cn+1)
cn

cn = x.find('l',cn+1)
cn

#원본문자열안에서 매개변수로 입력한 문자열이
#존재하는 위치를 앞에서부터 찾는다. 만약에 
#존재하지 않으면 error
x = 'hello world'
x.index('w')
x.index('a')#error가 나온다.

#원본문자열안에 매개변수로 입력한 문자열이
#몇 번 나오는지 껀수를 리턴하는 함수 
#R : stringr :: str_count(string, pattern)
x.count('l')

#원본문자열을 대문자로 변환하는 함수 #R : toupper()
x.upper()

#원본문자열을 소문자로 변환하는 함수 #R : tolower()
x.lower()

#첫글자만 대문자로 바꾸는 함수
x.capitalize()

#모든 단어의 첫글자를 대문자로 변환하는 함수 
x.title()

#대문자를 소문자로, 소문자를 대문자로 변환하는 함수 
s = 'HELLO world hi'
s.swapcase()

#원본 문자열을 지정한 공간에서 중앙에 배치하는 함수 
s.center(20)

#원본 문자열을 지정한 공간에서 왼쪽에 배치하는 함수 
s.ljust(20)

#원본 문자열을 지정한 공간에서 왼쪽에 배치하는 함수 
s.rjust(20)

x = '             hello               '
x
len(x)#33

#원본문자열에서 양쪽에 공백을 제거하는 함수
#R : stringr :: str_trim(x)
x.strip()

#원본문자열에서 왼쪽에 공백을 제거하는 함수
#R : str_trim(x, side = 'left')
x.lstrip()

#원본문자열에서 오른쪽에 공백을 제거하는 함수
#R : str_trim(x, side = 'right')
x.rstrip()

#원본문자열에서 문자를 제거하는 함수 
x = 'helloh'
x.strip('h')#'ello'
x.lstrip('h')#'elloh'
x.rstrip('h')#'hello'


x = 'hello'
y = 'hello2019'
z = '안녕하세요ㅎㅎ'
#원본 문자열이 숫자, 기호를 제외한 알파벳, 한글로
#이루어졌는지 확인하는 함수
x.isalpha()#True
y.isalpha()#False
z.isalpha()#True

#원본 문자열이 알파벳, 숫자로 이루어져있는지
#확인하는 함수 
x.isalnum()#True
y.isalnum()#True
z.isalnum()#True

# 원본 문자열이 숫자로만 이루어져있는지 확인하는 함수
x.isnumeric()#False
y.isnumeric()#False
z.isnumeric()#False

d = '2019'
d.isnumeric()#True
type(d)#str

#isnumeric() 은 문자함수임에 유의한다. 
#변수의 타입을 체크하는게 아니다.
e = 2019
e.isnumeric()#error
type(e) == int #True

#변수의 타입을 체크하려면 다음 함수를 이용한다.
isinstance(x, int)#False
isinstance(x, float)#False
isinstance(x, bool)#False
isinstance(x, str)#True

# in 연산자를 이용해서 문자열이 존재여부를 확인하는 연산자
x = 'hello world'
'w' in x

x == 'hello, world'

# 원본 문자열에서 매개변수로 입력한 문자열을 기준으로 원본문자열을 나눠
#리스트로 만드는 함수 
x = 'hello, world'
x.split(',')#['hello', ' world']
#R : strsplit(x,split = ','), stringr :: str_split(x, ',')

#원본 글자 사이에 특정한 문자열을 추가하는 함수 
x = 'abc'
','.join(x)#'a,b,c'
''.join(x)#R : stringr :: str_c()

#이것 말고도 문자함수는 많습니다.~~

#a = 'a b c d e f g' 변수에 문자열이 들어 있습니다. 다음을 수행하세요.
#[문제4] a 변수에 있는 문자의 갯수를  구하세요.
a = 'a b c d e f g'
len(a)

#[문제5] a 변수에 공백문자 갯수를 구하세요.
a.count(' ')

#[문제6] a 변수 안에 있는 공백문자를 제외한 갯수를 구하세요.
len(a) - a.count(' ')

#[문제7] a 변수에 있는 공백문자를 제거한 후 b 변수에 넣어주세요
b = a.replace(' ','')

#[문제8] a 변수에 있는 문자를 분리한 후 c 변수에 넣어주세요.
c = a.split(' ')
c
#[문제9] c 변수에 있는 문자를 합쳐서 자신의 변수에 다시 넣어주세요.
c = ''.join(c)
c = ','.join(c)
c

#아래와 같은 문자데이터가 있습니다. 
url = 'http://www.python.org'
#[문제10] http:// 제거한 후 url 변수에 넣어 주세요.
url = url.replace('http://','')
url.lstrip('http://')#이건 맨 왼쪽에 있는것만 제거 
#http:// 가 여러군데 나오면 위의 두 함수의 결과가
#달라질 것이다.
url
#[문제11] url변수에 있는 문자 데이터에 '.'을 기준으로 분리하세요.
url = url.split('.')
url#['www', 'python', 'org']

#[문제12] url변수에 있는 문자데이터를 www.python.org 모양으로 만드세요.
url = '.'.join(url)
url#'www.python.org'
#[문제13] url변수에 있는 문자데이터를 대문자로 변환하세요.
url = url.upper()
url
#[문제14] url변수에 있는 문자데이터를 소문자로 변환하세요.
url = url.lower()
url

