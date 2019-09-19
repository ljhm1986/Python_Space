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

#문자 * 숫자  = 문자가 숫자만큼 반복됨
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

#슬라이싱 #R : [시작요소번호:종료요소번호]
x[:]
x[0:5] #[시작요소번호:종료요소번호(이전까지)]
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
#원본 문자열이 매개변수로 입력한 문자열로 시작되는지 판단하는 함수
x.startswith('h')
x.startswith('H')

#원본 문자열이 매개변수로 입력한 문자열로 끝나는지 판단하는 함수
x.endswith('ld')
x[-2:] == 'ld'

x.endswith('D')
x[-1] == 'D'
x[len(x)-1] == 'D'

#원본문자열안에서 매개변수로 입력한 문자열이 존재하는 위치를 
#앞에서부터 찾는다. 만약에 존재하지 않으면 -1로 나온다.
#R : stringr :: str_replace(x,'w')
x.find('w')#w가 처음 나오는 위치
x.find('world')#world가 처음 나오는 위치
x.find('W')#-1

#문자열안에 l 이 여러개 있으면 다음처럼 반복한다.
cn = x.find('l')
cn
cn = x.find('l',cn+1)
cn
cn = x.find('l',cn+1)
cn

#원본문자열안에서 매개변수로 입력한 문자열이 존재하는 위치를 
#앞에서부터 찾는다. 만약에 존재하지 않으면 error
x = 'hello world'
x.index('w')
x.index('a')#error가 나온다.

#원본문자열안에 매개변수로 입력한 문자열이 몇 번 나오는지 껀수를 리턴하는 함수 
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

#원본 문자열을 지정한 공간에서 오른쪽에 배치하는 함수 
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
''.join(x)#abc #R : stringr :: str_c()

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
#http:// 가 여러군데 나오면 위의 두 함수의 결과가 달라질 것이다.
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

##############################################################################
#9/3#
#[문제 15] 반복문을 사용하지 않고 * 를 가로 10개 출력
print('*'*10)
#[문제 16] 반복문을 사용하지 않고 * 를 세로 10개 출력
print('*\n'*10)
#[문제 17] day 변수에 20190903을 입력하세요.
#화면 출력은  2019년 09월 03일 출력하세요.
day = '20190903'
print(day[0:4]+'년 '+day[4:6]+'월 '+day[6:]+'일')
print('{}년 {}월 {}일'.format(day[0:4],day[4:6],day[6:8]))
print('%s년 %s월 %s일'%(day[0:4],day[4:6],day[6:8]))
day2 = 20190903
divmod(day2,10000)


#python 자료형
#1. list
#- 서로 다른 데이터 타입을 갖을 수 있는 배열타입
#- 중첩할 수 있다. [[],[]]...
#- [], list() 로 표현한다.
#R : list(key = value, key = value)

x = [] #R <- NULL
type(x)#list

y = list()
type(y)#list

x = [10, 20, 30]
print(x)
type(x)#list
len(x)#3

# list index
x[0]#10
x[-1]#30

isinstance(x[0],int)#True
isinstance(x[0],float)#False
isinstance(x[0],str)#False

#list 슬라이싱 [시작:끝-1]
x[0:]
x[0:2]
x[:-1]
x[-1:]

#list 값 수정하는 방법
#R : x[[0]] <- 100, x$기존의 key <- 100
x[0] = 100
x
x[1:3] = [200,300]

#list variable 끝에 값을 추가해 넣기
#R : x$새로운 key <- 400
x.append(400) 
x#[100, 200, 300, 400]

#기존 리스트 변수에 다른 리스트를 이어 붙일 때는 extend를 사용한다.
x = [100, 200, 300, 400]
x1 = [600, 700, 900]

x.extend(x1)
x#[100, 200, 300, 400, 600, 700, 900]

#또는 + 로도 리스트 변수를 결합할 수 있다.
x = x + x1
x#[100, 200, 300, 400, 600, 700, 900]

#append를 사용하면 리스트가 중첩되어서 추가된다.
x.append(x1)
x#[100, 200, 300, 400, [600, 700, 900]]
x[-1]#[600, 700, 900]

#삭제하기
del x[-1]
x#[100, 200, 300, 400]

#x#[100, 200, 300, 400, 600, 700, 900]
x[4]#600
#인덱스를 사용하여 특정 위치에 값을 입력하는 방법
x.insert(4,500)
x[4]#500
x#[100, 200, 300, 400, 500, 600, 700, 900]

x[-1]#900
#리스트 변수에 있는 값 중에 마지막 값을 제거하는 방법
x.pop()
x#[100, 200, 300, 400, 500, 600, 700]

#특정한 인덱스 번호를 기준으로 제거하는 방법
#R : x[[4]] <- NULL, x$기존의 key <- NULL
x.pop(4)
x#[100, 200, 300, 400, 600, 700]

del x[3]
x#[100, 200, 300, 600, 700]

#변수를 제거하는 방
del x
x#error
dir()

drink = ['콜라','사이다','콜라','환타','사이다','콜라']
len(drink)#6
drink.count('콜라')#3
drink.find('콜라')#error #find는 list일때는 사용할 수 없다.
drink.index('콜라')#0
drink.index('콜라',1)#2
drink.index('콜라',3)

#list 변수에 값을 기준으로 삭제하는 방법
drink.remove('콜라')
drink#['사이다', '콜라', '환타', '사이다', '콜라'] #하나만 지워진다.

#중첩 리스트
x = [1,2,3,['a','b','c'],4,5]
type(x)
type(x[0])#int
type(x[3])#list
x[3][0]#'a'
type(x[3][0])#str
x[3][0:2]#['a', 'b']
x[3][-1]#'c'
x[3].append('d')
del x[3][-1]
x[3][0] = x[3][0].upper()
x[0] = x[0] * 10
x#[10, 2, 3, ['A', 'b', 'c'], 4, 5]

x[3][0] = x[3][0] * 2
x#[10, 2, 3, ['AA', 'b', 'c'], 4, 5]

#list 변수 값을 지우는 방법
x.clear()
x#[]
type(x)#list
dir()#x 있음 
id(x)#메모리 주소값 

del x
dir()
id(x)#error

x = [1,5,3,4,2]
#리스트 변수안에 값을 기준으로 정렬(기본값은 오름차순)
x.sort()#'미리보기 아님' 에 주의하자, x값이 달라진다.
x#[1, 2, 3, 4, 5]

x.sort(reverse=True)
x#[5, 4, 3, 2, 1]

#sort 를 미리보기만 하려면 sorted()를 사용한다.
x = [1,5,3,4,2]

sorted(x)#[1, 2, 3, 4, 5]
x#[1,5,3,4,2]
sorted(x, reverse = True)#[5, 4, 3, 2, 1]

#리스트 인덱스 순서를 반대로 뒤집을때 사용하는 방법
x[::-1]#[2, 4, 3, 5, 1]
#
x.reverse()
x#[2, 4, 3, 5, 1]


king =[['주몽','유리왕','대무신왕'],['태종무열왕','경덕왕'],
	['온조왕','개로왕']]
len(king)#3
len(king[1])
#[문제18] 1번 인덱스에 '원성왕' 추가하세요.
king[1].append('원성왕')
king
#[문제19] 2번 인덱스에 '무왕' 추가하세요.
king[2].append('무왕')
king
#[문제20] 0번 인덱스에 '대무신왕' 추가하세요.
king[0].append('대무신왕')
king
#[문제21] 0번 인덱스에 '미천왕','미천왕','광개토대왕' 한꺼번에 추가하세요.
king[0].extend(['미천왕','미천왕','광개토대왕'])
king
#[문제22] 2번 인덱스에 2번 위치에 '성왕' 추가하세요.
king[2].insert(2,'성왕')
king
#[문제23] '미천왕' 갯수를 세어주세요.
king[0].count('미천왕')#2
#[문제24] 0번 인덱스에 마지막 데이터를 삭제해주세요.
king[0].pop()
king

#list의 원소들을 정렬하기 
king[0].sort()
king[2].sort()
king.sort()#이것도 정렬이 되는데, 기준이 각 list[0] 인가?
king

## 2.tuple(튜플) ##
#-리스트와 유사하게 원소들을 나열한다. 다른점은 수정, 삭제, 추가를 할 수 없다.
#-상수와 비슷하다.
#-() 
lst = []
type(lst)#list

tuple1 = ()
type(tuple1)#tuple

#()를 사용하지 않고 tuple을 선언할 수 있다.
tuple2 = 10, 20
tuple2
type(tuple2)#tuple

x = (1,)
type(x)#tuple

tuple3 = (1,2,3,4,5)
tuple4 = ('a','b',('ab','ac'))#tuple도 중첩이 가능하다.
type(tuple4[2])#tuple

tuple5 = (1,2,('a','b','c'),3,(1,2,3))
type(tuple5[0])#int
type(tuple5[2][0])#str

#tuple 2개를 합치
x = (1,2,3)
y = (4,5,6)

c = x + y
type(c)#tuple
c#(1, 2, 3, 4, 5, 6)
#tuple은 수정, 삭제, 추가를 할 수 없다.
c[10] = 10#error
del c[0]#error
c.append(7)#error

c.index(1)
c.count(1)

#()없이도 tuple을 선언할 수 있다.
a = 1,2,3
type(a)#tuple
#tuple 안의 값들이 각각 분리되어 들어간다.
one, two, three = a
one#1
two#2
three#3

#다만 원소의 갯수와 분배하려는 요소들의 갯수가 같아야 한다.
ga, na = a
ga, na, da, la = a

#리스트도 가능하다.
a = [1,2,3]
type(a)
#각각으로 분리되어 들어간다.
one, two, three = a
one#1
two#2
three#3

#슬라이싱을 통해서 각각의 변수에 값을 넣을 수 있다.
x , y = a[0:2]
x
y

#수정, 삭제, 추가등의 기능이 없기 때문에 튜플이 리스트보다 처리속도가 빠르다.

## 3.dictionary ##
#- key, value 값을 가지는 자료형이다.
#R : list가 key, value 있었다.
#이름 = 홍길동, 전화번호 = 010-0000-1111, 주소 = 서울

dic = {}
dic
type(dic)#dict

dic = {'name'  : '홍길동',
       'phone' : '010-0000-1111',
       'addr'  : '서울시'}

dic#{'name': '홍길동', 'phone': '010-0000-1111', 'addr': '서울시'}

sports = {'축구':'메시','농구':'커리','야구':'박찬호'}

sports['축구']#'메시'
sports['농구']#'커리'
sports['컬링'] = '김영미'
sports
#{'축구': '메시', '농구': '커리', '야구': '박찬호', '컬링': '김영미'}
sports['컬링'] = ['김은정','김경애','김영미']
sports
#{'축구': '메시', '농구': '커리', '야구': '박찬호',
# '컬링': ['김은정', '김경애', '김영미']}

sports.keys()#dict_keys(['축구', '농구', '야구', '컬링'])
sports.values()#dict_values(['메시', '커리', '박찬호', ['김은정', '김경애', '김영미']])
sports.items()#(key, value) 식으로 묶겨서 나옴

# key값에 대한 value 값 확인
sports['농구']#'커리'
sports.get('농구')#'커리'

sports['골프']#error
sports.get('골프')#키값이 없으면 아무것도 안 나옴 

'골프' in sports.keys()#False
'농구' in sports.keys()#True
#R : %in%

#값이 있으면 True, 없으면 False
'커리' in sports.values()
'김영미' in sports.values()#False!!!
['김은정','김경애','김영미'] in sports.values()#True

#키, 값을 삭제
del sports['야구']
sports

#값만 삭제하려면 
sports['축구'] = []
sports

#다시 값을 넣어보자 
sports['축구'] = '손흥민'
sports

#변수안에 키, 값 내용 지우기
sports.clear()
sports#{}

#dictionary 형은 wordcloud등을 이용하려고 단어수 빈도를 측정할때
#사용한다.

sports = {'축구':'박지성','농구':'조던'}

v = sports.values()
v
type(v)#dict_values
list(v)#['박지성', '조던']

w = sports.keys()
w
type(w)#dict_keys
list(w)#['축구', '농구']

k, v = sports.items()
k#('축구', '박지성')
v#('농구', '조던')

## 4.set ##
#- 집합, 중복을 허용하지 않습니다.
#- 리스트와 비슷하다. 인덱스의 순서가 없다.
#선언은 {} 사용하는데 key, value 는 없다.
s = {1,1,1,1,5,76,3,3,4,6,56,2,3,4,54,3,57,76,42,4,4,6,7}
s#{1, 2, 3, 4, 5, 6, 7, 42, 54, 56, 57, 76}
type(s)#set

x = {1,2,3,5}
y = {1,2,3,4,6}
x
y

#합집합 #R : union(x,y)
x.union(y)#{1, 2, 3, 4, 5, 6}
x|y#{1, 2, 3, 4, 5, 6}

#교집합 #R : intersect(x,y)
x.intersection(y)#{1, 2, 3}
x&y#{1, 2, 3}

#차집합 #R : setdiff(x,y)
x.difference(y)#{5}
x-y#{5}

y.difference(x)#{4, 6}
y-x#{4, 6}

#x나 y에만 포함되는 원소들은?
x.difference(y).union(y.difference(x))#{4, 5, 6}
x.symmetric_difference(y)#{4, 5, 6}

#원소가 set안에 있는지 보자 
1 in x#True
7 in x#False
44 in s#False

#set은 index가 없다.
s[0]#error

# 집합변수의 값을 삭제
x.remove(1)
x#{2, 3, 5}

# 집합변수의 값을 추가
x.add(1)
x.add(9)
x
#여러개의 값을 추가
x.add([22,33])#error
x.update([22,33])
x#{1, 2, 3, 5, 9, 22, 33}

##
x = []
y = ()
z = {}
type(x)#list
type(y)#tuple
type(z)#dict

s = set()
type(s)#set
id(s)#메모리 할당주소 : 149245736

## 5.bool ##
#참(True), 거짓(False)을 나타내는 자료형 

x = True
y = False
type(x)
type(y)

1 == 1
x == y
1 > 2
2 >= 1
1 <= 1
1 < 2
1 != 2

# True 표현하는 방법
bool(1)
bool(-1)
bool('이재훈')
bool('python')
bool([1,2,3])
bool((1,2,3))
bool({1,2,3})
not 0
not None

#False 표현방법
bool(0)
bool([])
bool(())
bool({})
bool(None)
bool('')
not 1
not -1


#복제를 할때 주의해야 할 점 
a = [1,2,3]
b = a
b#[1, 2, 3]

#메모리 위치는 같다. 새롭게 만든게 아니라 참조만 같이 해 놓았다.
id(a)#123758152
id(b)#123758152

a[0] = 10
a#[10, 2, 3]
b#[10, 2, 3]
#값이 같이 바뀌는걸  알 수 있다. #따로 값을 바꾸는 방법은?
#따로 분리해서 사용하는 방법은?

import copy
c = copy.deepcopy(a)
#다음 값이 다른게 확인된다.
id(a)
id(c)

a.append(4)
a#[10, 2, 3, 4]
b#[10, 2, 3, 4]
c#[10, 2, 3]

#
x = [1,2,3]
y = x[:]
id(x)#123944456
id(y)#123934920
x.append(4)
x#[1, 2, 3, 4]
y#[1, 2, 3]

############################################################################
#9/4#

#input() 함수를 이용해서 사용자로 부터 입력을 
#받아들여 프로그램에게 전달해주는데, 이때 전달되는 데이터의
#형식은 문자열이다.

x = input()#하고 console로 가서 값을 입력한다.
y = input()
x#'1000'
y#'20'
type(x)
type(y)
x + y#'100020'
#숫자로 계산하고 싶으면 형변환을 한 뒤에 해야 한다.
int(x) + int(y)

x1 = input()
x2 = input()

int(x1) + int(x2)#error
float(x1) + float(x2)

#문구도 같이 나오게 하고 싶을때
x = input('숫자를 입력하세요 :')
#처음부터 형변환을 해서 입력되게 하자
x = int(input('정수를 입력하세요:'))

x

#조건제어문
#1. if문
#if 조건문 : 
#   수행해야할 문장

#if 조건문 :
#   수행해야할 문장
#else:
#   수행해야할 문장

x = 10
if x == 10:
    print('x는 10이다')
    print('오늘 하루도 행복하자')

x = 20
if x == 10:
    print('x는 10이다')
    print('오늘 하루도 행복하자')
else:
    print('x는 10아니다')
    print('오늘 하루도 {}배 행복하자'.format(x))
    
x = 1
if x:
    print('참')
else:
    print('거짓')
#참 

x = 0
if x:
    print('참')
else:
    print('거짓')
#거짓
    
x = '파이썬'
if x:
    print('참')
else:
    print('거짓')
#참

x = []
if x:
    print('참')
else:
    print('거짓')
#거짓

#주의할 점
x = 0
if x > 10 and 1/x:
    print('x는 10보다 크다')
else:
    print('x는 10보다 작다')
#and는 앞에 false이면 뒤에 안 보고 false로 판별한다
#그래서 에러가 나지 않고 작동한다.
    
x = 0
if x > 10 & 1/x:
    print('x는 10보다 크다')
else:
    print('x는 10보다 작다')
#&는 뒤에도 보기 때문에 error:ZeroDivisionError: division by zero
    
#
x = 0
if x < 10 or 1/x:
    print('x는 10보다 작다')
else:
    print('x는 10보다 크다')
#or 역시 앞에만 true 인거 보고 작동한다.   
x = 0
if x < 10 | 1/x:
    print('x는 10보다 작다')
else:
    print('x는 10보다 크다')
#|는 뒤에도 보기 때문에 error:ZeroDivisionError: division by zero
    
num = int(input('점수를 입력해주세요 :'))

if 90 <= num <= 10000:
    grade = 'A'
elif 80 <= num < 90:
    grade = 'B'
elif 70 <= num < 80:
    grade = 'C'
elif 60 <= num < 70:
    grade = 'D'
else:
    grade = 'F'

print('학점 : ' + grade)

#[문제 25] 숫자를 입력값으로 받아서 그 값이 짝수면 짝수,
#홀수면 홀수를 출력해주세요.
x = int(input('정수를 입력하세요 :'))

if x <= 0 :
    print('자연수가 아닙니다.')
elif x%2 == 0:
    print('짝수입니다.')
elif x%2 == 1:
    print('홀수입니다.')
    
#[문제 26] 한글, alphabet만 입력받아서 그 문자를 출력하고
#아니면 '다른 문자가 포함되어 있습니다' 출력해주세요 
y = input("입력 :")
    
y.replace('[가-힣]','')#안되네 
y.replace('[a-z]','')#안되네 

if y.isalpha():
    print(y)
else:
    print('다른 문자가 포함되어 있습니다.')

#[문제 27] 숫자를 입력값으로 받아서 숫자면 입력받은
#숫자를 출력하고 아니면 입력값이 들어 오지 않았으면
#"입력값이 없습니다" 출력하고 아니면"숫자이외의 문자가
#포함되어 있습니다" 를 출력해주세요
    
x = input('숫자를 입력하세요 : ')
#아무것도 입력하지 않고 엔터만 친 경우
x#''
x == None#False
#'' 는 None 이 아니다.

if x.isnumeric():
    print(x)
else:
    if x == '':
        print('입력값이 없습니다.')
    else:
        print('숫자이외의 문자가 포함되어 있습니다.')
        
x = None
type(x)#NoneType ,무슨 타입인지 모르겟다
x == ''#False

y = ''
type(y)#str

if x == None:
    print('값이 없습니다.')
else:
    print(x)

if x is None:
    print('값이 없습니다.')
else:
    print(x)
    
x = 1
if isinstance(x, float):
    print(x)
else:
    print('float형으로 입력하세요')
    
x = 1.4
if isinstance(x, int):
    print(x)
else:
    print('int형으로 입력하세요')
    
#list는 인덱스끼리 비교합니다.
x = [1,2,3]
y = [2,1,3]

if x==y:
    print('참')
else:
    print('거짓')

#set일 때는 원소가 다 같으면 된다.
x = {1,2,3}
y = {2,1,3}

if x==y:
    print('참')
else:
    print('거짓')

sal = 1000
comm = None
sal * comm#error

#한줄에 if 구조를 표현하는 방식 
#(if)리턴값 if 조건 else (else)리턴값
annual = sal if comm is None else sal*comm 
print(annual)


sal = 1000
comm = 1.1
annual = sal if comm is None else sal*comm 
print(annual)

#반복문
##1. while ##
#   조건이 True인 동안에 반복 수행한다.

#while 조건문:
#    반복수행할 문장

i = 0
while i <= 10:
    print(i)
    i = i + 1
    
#[문제 28] 1부터 100까지 합을 구하세요.

i = 1
sum = 0
while i <= 100:
    sum = sum + i
    i = i + 1
print(sum)

i = 1
sum = 0
while i <= 100:
    sum = sum + i
    if i%10 == 0:
        print(sum)
    i = i + 1
    
#break : 반복문을 중단한다.
while True:
    print('행복하자')
    break

while True:
    answer = input('반복할까요?[Y/N] : ')
    if answer.upper() == 'Y':
        print('반복을 계속합니다.')
    elif answer.upper() == 'N':
        print('반복을 종료합니다.')
        break
    else:
        print('Y/N 입력하세요')
        
#[문제 29] 1부터 100까지 3의 배수를 출력하고
#합도 구하세요
i = 1
sum = 0
while i <= 100:
    if i%3 == 0:
        print(i)
        sum = sum +i
    i = i + 1
print(sum)

i=0
sum = 0
while i <100:
    i += 3
    if i > 100:
        break
    else:
        sum += i
        print(i)
print(sum)

#[문제 30] 1부터 10까지 홀수값만 출력해주세요
i = 0
while i <= 10:
    if i % 2 == 1:
        print(i)
    i = i + 1

#continue 하면 밑에 문장을 실행하지 않고 
#바로 다음 반복부분으로 간다.
i = 0
while True:
    i = i + 1
    if i == 10:
        break
    else:
        if i % 2 == 0:
            continue
    print(i)

#[문제 31] 구구단을 입력값으로 받아서 구구단을 출력해주세요

x = int(input('2~9 사이 숫자를 입력하세요 :'))

i = 1
while i < 10:
    print('%d * %d = %d'%(x,i,x*i))
    i = i + 1


#[문제 32] 구구단을 출력해주세요
i = 2
j = 1
while i < 10:
    while j < 10:
        print('%d * %d = %d'%(i,j,i*j))
        j = j + 1
    j = 1
    i = i + 1

dan = 2
while dan <= 9:
    i = 1
    print("{} {}단 {}".format('*'*2, dan, '*'*2))
    while i <= 9:
        print("{} * {} = {}".format(dan, i, dan*i))
        i += 1
    dan += 1
    
#[문제 33] 2단을 가로로 출력해주세요
#2 * 1 = 2 2 * 2 = 4 ... 2 * 9 = 18
    
dan = 2
i = 1
string = ''
while  i <= 9:
    string = string + (str(dan) + ' * ' + str(i) +' = '+ str(dan*i).ljust(5))
    #ljust() 는 문자일때만 사용이 가능하다. 문자형으로 바꾸고 해야 한다.
    i = i + 1
print(string)

#[문제 34] 구구단을 가로로 출력해주세요
#2 * 1 = 2  3 * 1 = 3  4 * 1 = 4 ...
#2 * 2 = 4  3 * 2 = 6  4 * 2 = 8 ...

i = 1
while i <= 9:
    dan = 2
    string = ''
    while dan <= 9:
        string = string + '{} * {} = {}'.format(dan, i, str(dan*i).ljust(5))
        dan = dan + 1 
    print(string)
    i = i + 1

i = 1
string = ''
while i <= 9:
    dan = 2
    while dan <= 9:
        string = string + '{} * {} = {}'.format(dan, i, str(dan*i).ljust(5))
        dan = dan + 1 
    string = string + '\n'
    i = i + 1
print(string)


##2. for문 ##
#for 변수 in (리스트, 튜플, 문자열):
#   수행해야할 문장

#list 
x = ['sql','plsql','R','Python']
for i in x:
    print(i)
 
#문자열
for i in '오늘 하루도 행복하자':
    print(i)

#tuple
x = ('sql','plsql','R','Python')
for i in x:
    print(i)

#list안에 tuple
x = [(1,2),(3,4),(5,6)]
type(x)
type(x[0])

for i in x:
    print(i)
#(1, 2)
#(3, 4)
#(5, 6)
    
for (a,b) in x:
    print(a,b)
#1 2
#3 4
#5 6
    
for (a,b) in x:
    print(a+b)

#[문제 35] 학생들의 점수가 90, 55, 63, 78, 80 점이다.
#60점 이상이면 합격, 60점 미만이면 불합격을 출력해주세요
score = [90, 55, 63 , 78, 80]
for i in score:
    if i >= 60:
        print('합격')
    else:
        print('불합격')

##############################################################
#9/5#
#range(시작점, 끝점 + 1,증가분:기본값 = 1)
        
list(range(1,101))#1부터 100까지 포함한 list

for i in range(1,11):
    print(i)
    
#[문제 36] 1부터 100까지 합을 구하세요. for문 이용
sum = 0
for i in range(1,101):
    sum = sum + i
print(sum)

#[문제 37] 1부터 100까지 3의 배수이면서 동시에 
#5의 배수인 수를 출력해주세요.
for i in range(1,101):
    if i % 3 == 0 and i % 5 == 0:
        print(i)
        
#연산자 연산 순위가 &이 == 보다 높아서 ()해야한다.
for i in range(1,101):
    if (i % 3 == 0) & (i % 5 == 0):
        print(i)
        
for i in range(1,101):
    if i % 3 == 0:
        if i % 5 == 0:
            print(i)
            

#[문제 38] 1부터 10까지 4,8은 제외시킨 수를 출력하세요
            
for i in range(1,11):
    if i != 4 and i != 8:
        print(i)

for i in range(1,11):
    if i == 4 or i == 8:
        continue
    print(i)
        
for i in range(1,11):
    if (i != 4) & (i != 8):
        print(i)

#[문제 39] 구구단을 출력해주세요       
for i in range(2,10):
    for j in range(1,10):
        print('%d * %d = %d'%(i,j,i*j))
        
for j in range(1,10):
    string = ''
    for i in range(2,10):
        string = string + str(i) + ' * ' + str(j) + ' = ' + str(i*j).ljust(5)
    print(string)

string = ''   
for j in range(1,10):    
    for i in range(2,10):
        string = string + str(i) + ' * ' + str(j) + ' = ' + str(i*j).ljust(5)
    string = string +'\n'
print(string)

#[문제 40] list변수에 a,b,c,d 값이 있습니다.
#for문을 이용하여 아래 화면과 같이 출력하세요.
#0번 a값이 있습니다.
#1번 b값이 있습니다.
#2번 c값이 있습니다.
#3번 d값이 있습니다.

lst = ['a','b','c','d']
for i in range(0,4):
    print('{}번 {}값이 있습니다.'.format(i,lst[i]))
    
lst = ('a','b','c','d')
for i in range(0,4):
    print('%d번 %s값이 있습니다.'%(i,lst[i]))
    
list(range(0,4))#[0, 1, 2, 3]
list(range(4))#[0, 1, 2, 3]
list(range(len(lst)))#[0, 1, 2, 3]
lst = ['a','b','c','d']
for i in range(len(lst)):
    print('{}번 {}값이 있습니다.'.format(i,lst[i]))
   

#enumerate() : index, 실제값을 리턴하는 함수 
enumerate(lst)#객체정보만 보이는데 실제로는 index와 값을 보내준다.
for idx, value in enumerate(lst):
    print('{}번 인덱스에 {}이 있습니다.'.format(idx,value))

 
lst = {'a','b','c','d'}
#본래 set은 index가 없다. enumerate를 사용하면 자동으로 index를 매겨서 출력한다.
for idx, value in enumerate(lst):
    print('{}번 인덱스에 {}이 있습니다.'.format(idx,value))
    
#[문제41] 리스트 변수에 18,2,3,1,4,5,7,8,9,10,11,15,16 값이 들어 있습니다.
# 짝수만 합을 구하세요. 
#1) while문
#2) for문
lst = [18,2,3,1,4,5,7,8,9,10,11,15,16]
sum = 0
i = 0
while i < len(lst):
    if lst[i] % 2 == 0:
        sum = sum + lst[i]
    i = i + 1
print(sum)

lst = [18,2,3,1,4,5,7,8,9,10,11,15,16]
sum = 0
i = 0
while i, v in enumerate(lst):#안되넹 
    if v % 2 == 0:
        sum = sum + v
    i = i + 1
print(sum)

lst = [18,2,3,1,4,5,7,8,9,10,11,15,16]
sum = 0
for value in lst:
    if value % 2 == 0:
        sum = sum + value
print(sum)
    
#[문제42] 과일의 빈도수를 만들어 주세요.
#(앞으로도 빈도수 체크를 할때 사용할 수 있습니다. 예) wordcloud를 사용할때)
#fruit = ("사과","귤","오렌지","배","포도","바나나","키위","딸기","블루베리",
# "망고","수박","사과","귤","키위","포도","바나나","사과","딸기","블루베리",
# "망고","사과","귤","오렌지","배","포도","바나나","사과","딸기","파인애플")
#{'사과': 5,
# '귤': 3,
# '오렌지': 2,
# '배': 2,
# '포도': 3,
# '바나나': 3,
# '키위': 2,
# '딸기': 3,
# '블루베리': 2,
# '망고': 2,
# '수박': 1,
# '파인애플': 1}
fruit = ("사과","귤","오렌지","배","포도","바나나","키위","딸기","블루베리",
 "망고","수박","사과","귤","키위","포도","바나나","사과","딸기","블루베리",
 "망고","사과","귤","오렌지","배","포도","바나나","사과","딸기","파인애플")

dic = {}
type(dic)
for value in fruit:
    #print(value)
    if value in dic.keys():
        dic[value] = dic[value] + 1
    else:
        dic[value] = 1
print(dic)

dic.keys()
dic.values()
dic.items()

dic
sorted(dic)#key 기준으로 정렬됨, value들은 안 나옴
dic.items()

for key, value in dic.items():
    print(key, value)

#key 기준으로 정렬되어 출력된다.    
for key, value in sorted(dic.items()):
    print(key, value)
    
#내림차순
for key, value in sorted(dic.items(),reverse = True):
    print(key, value)
    
#value 기준으로 정렬하려면
import operator
#key 기준
for key, value in sorted(dic.items(),key = operator.itemgetter(0)):
    print(key, value)
#value 기준
for key, value in sorted(dic.items(),key = operator.itemgetter(1)):
    print(key, value)
#value 기준, 내림차순
for key, value in sorted(dic.items(),
                         key = operator.itemgetter(1),
                         reverse = True):
    print(key, value)
    
#[문제 43] x 리스트 변수에 1부터 10까지 입력하세요
#y변수는 x 변수의 값을 2곱한 값으로 입력해주세요

x = [1,2,3,4,5,6,7,8,9,10]
y = []
for i in range(len(x)):
    y.append(x[i]*2)
    
x = range(1,11)
y = []
for i in x:
    y.append(i*2)
print(y)

#list 내장 객체 
x = range(1,11)
z = [i*2 for i in x]
print(z)

#[문제 44] x list 변수에 apple, banana, orange 입력하시고
#이 값들의 문자의 길이를 출럭해 주세요
x = ['apple','banana','orange']
y = [len(i) for i in x]
print(y)    

#[문제 45] 
lst1 = [1,2,3]
lst2 = [4,5,6]
#결과는 
#4,5,6,8,10,12,12,15,18
for i in lst1:
    for j in lst2:
        print(i*j)
        
z = [i*j for j in lst1 for i in lst2]
print(z)

#가로방향으로 한 번에 출력하려면 print에 end를 넣는다.
for i in lst1:
    for j in lst2:
        print(i*j, end = ',')

lst3 = []
for i in lst1:
    for j in lst2:
        lst3.append(i*j)
print(lst3)

[i*j for i in lst1 for j in lst2]

#[문제 46] 1부터 100까지 짝수만 x변수에 입력하기
x = []
for i in range(1,101):
    if i % 2 == 0:
        x.append(i)
print(x)

x = [i for i in range(1,101) if i % 2 == 0]
print(x)

#[문제 47] tuple 변수에 사과, 귤, 오렌지, 배, 포도, 바나나, 자몽,
#키위, 딸기, 블루베리, 망고를 입력하시고 과일이름중에 세글자 이상인
#과일만 fruit_lst 변수에 입력해주세요

tup = ('사과','귤','오렌지','배','포도','바나나','자몽',
       '키위','딸기','블루베리','망고')

fruit_lst = []
for i in tup:
    if len(i) >= 3:
        fruit_lst.append(i)
print(fruit_lst)

fruit_lst = []
fruit_lst = [i for i in tup if len(i) >= 3]
print(fruit_lst)

#[문제 48] 과일판매 현황을 dictionary 변수로 생성하세요
#과일이름은 키로 하고 수량은 값으로 표현한 후 과일 이름만
#대문자로 출력해주세요
#apple 100, banana 300, orange 300
dic = {'apple' : 100, 'banana' : 300, 'orange' : 300}

for key in dic.keys():
    print(key.upper(), end = ',')
    
z = [key.upper() for key in dic.keys()]
print(z)

#[문제 49] 변수에 2, -1, 4, -10, 5, -9가 있습니다.
#음수값만 negative 변수에 입력해주세요
lst = [2, -1, 4, -10, 5, -9]
negative = []
for i in lst:
    if i < 0:
        negative.append(i)
print(negative)

negative = []
negative = [i for i in lst if i < 0]
print(negative)

#[문제 50] x변수의 있는 값을 y변수에 아래와 같이 저장하세요
#x = [2,-1,4,-10,5,-9]
#y = [2,'음수',4,'음수',5,'음수']
x = [2,-1,4,-10,5,-9]
y = []
for i in x:
    if i < 0:
        y.append('음수')
    else:
        y.append(i)
print(y)

z = []
#모두 출력해야 하니까 if else 가 앞에 온다.
z = [i if i > 0 else '음수' for i in x ]
print(z)

##########################################################
#9/6#
#함수
#기능의 프로그램
#반복되는 코드를 하나로 묶어서 처리하는 방법
#def 함수이름(형식매개변수, 인수, 인수):
#       수행할 문장
#       [return 값]
#함수이름(인수, 인수)

def message():
    print('오늘 하루도 행복하게 살자')
message()

def message(arg1):
    print('오늘 하루도 {}배 행복하게 살자'.format(arg1))
    return "happy"*arg1
word = message(100)
word

#[문제 51] 함수에 두개의 숫자를 인수값으로 받아서 값을 
#비교하는 함수를 생성하세요
#num_compare(10,20)
#10은 20보다 작다

def num_compare(int1, int2):
    if (int1 > int2):
        print('{}은 {}보다 크다.'.format(int1,int2))
    elif (int1 == int2):
        print('%d는 %d와 같다.'%(int1, int2))
    else:
        print('%f는 %f보다 작다'%(int1, int2))
num_compare(11.9,20)

#[문제 52] 두 인수값을 받아서 합한 값을 리턴하는
#sum  함수를 생성하세요
def sum1(int1, int2):
    return int1 + int2

sum1(2,3)

#입력변수의 갯수가 달라지는 경우, 앞에 * 붙인다.
def sum2(*arg):
    total = 0
    for i in arg:
        total = total + i
    return total

sum2(1,2,3,4,5)
sum2(5,6,3,5,6,7,3)

#[문제 53] cal 함수를 생성하세요
#def 함수이름( 인수1, *인수2):
#       수행할 문장

def cal(char, *arg):
    if char.lower() == 'sum':
        total = 0
        for i in arg:
            total = total + i
        return total
    elif char.lower() == 'multiply':
        total = 1
        for i in arg:
            total = total*i
        return total
    else:
        print('sum 또는 multiply중에서 입력해 주세요.')

cal('sum',1,2,3,4,5)
cal('multiply',1,2,3,4,5)
cal('minus',1,2,3,4)

#[문제 54] 여러숫자를 인수값으로 받아서 합과 평균을 
#출력하는 aggF 함수를 생성하세요
#aggF(1,2,3,4,5,6,7,8,9,10)
#합 : 55
#평균 : 5.5

def aggF(*arg):
    sum = 0
    mean = 0
    for i in arg:
        sum = sum + i
    print('합 : {}'.format(sum))
    mean = sum/len(arg)
    print('평균 : {}'.format(mean))

aggF(1,2,3,4,5)
aggF(1,2,3,4,5,6,7,8,9,10)

#return 2개 사용?을 해 보자 
def f1(x,y):
    return x + y
    return x * y

f1(2,3)#위에 return 하나만 나온다.

#다음과 같이 바꾸어서 해 보자 
def f1(x,y):
    return x + y, x * y

sum, mul = f1(2,3)
print(sum)
print(mul)

#return 을 만나면 바로 종료한다.
def f2(x,y):
    if y == 0:
        return 
    print(x/y, "값 입니다.")
    
f2(4,2)
f2(4,0)

#값이 들어오지 않을 경우를 대비해서 default 설정
def f3(name, age, gender = "M"):
    print('이름은 ',name)
    print('나이는',age)
    if gender == 'M':
        print('남자')
    else:
        print('여자')
        
f3("홍길동",20)
f3('이재훈',25)
f3('유관순',20,'F')

#전역변수(global) : 프로그램이 종료될 때까지 
#어디서든지 사용할 수 있는 변수
#지역변수(local) : 함수안에서만 사용하는 변수
x = 10
def f4(arg):
    print('x변수 값은', arg)
    x = 20
    print('x변수 값은', x)

f4(x)
print(x)#10

#함수 안에 global 넣으면 
x = 10
def f5(arg):
    print('x변수 값은', arg)
    global x
    x = 20
    print('x변수 값은', x)

f5(x)
print(x)#20

#[문제 55] 입력값을 더하는 함수를 구하세요
#print(add(2))
#2
#print(add(8))
#10
y = 0
def add(x):
    global y   
    y = y + x
    return y

print(add(3))
print(add(8))

dir()

#[문제 56] 아래와 같이 변수에 값이 들어 있습니다.
#exchange함수에 x변수에 값을 넣으면 y로 변환하는
#함수를 생성하세요
x = ['귀도','반','로섬']
y = ['Guido','van','Rossum']

#함수안에서 복사할 때 주의할 점이 있다.
def exchange(input):
    x = input #동일한 메모리의 x 변수를 사용한다.
    #index 해서 값을 넣으면 global 처럼 원래 x변수가 바뀐다.
    for i in range(len(x)):
        x[i] = y[i]
    print(x)
    
exchange(x)
print(x)    
        
#다음과 같이 바꾸자 
def exchange2(input):
    x = input[:]
    for i in range(len(x)):
        x[i] = y[i]
    print(x)

exchange2(x)
print(x)

#또는
import copy

def exchange3(input):
    x = copy.deepcopy(input)
    for i in range(len(x)):
        x[i] = y[i]
    print(x)

exchange3(x)
print(x)

#다음과 같이 해도 x가 바뀐다. 주의하자 
x = ['귀도','반','로섬']
y = ['Guido','van','Rossum']

def exchange4(input):    
    for i in range(len(input)):
        input[i] = y[i]
    print(input)
    
exchange4(x)
print(x)   

#[문제 57] 약수를 구하는 divisor 함수를 생성하세요

def divisor():
    x = input('입력하세요 :')
    if not int(x):
        print('자연수를 입력하세요')
    else:
        x = int(x)
        if x <= 0:
            print('자연수를 입력하세요')
        else:
            for i in range(1,x+1):
                if x % i == 0:
                    print(i, end = ' ')

def divisor2(x):
    if not isinstance(x, int):
        print('자연수를 입력하세요')
    else:
        y = int(x**(1/2))
        for i in range(1,y+1):
            if x % i == 0:
                print(i, end = ' ')
                if i != int(x/i):
                    print(int(x/i), end = ' ')

divisor()

def divisor3(x):
    num = []
    for i in range(1,int(x/2)+1):
        if x % i == 0:
            num.append(i)
    num.append(x)
    return num
    
divisor3(90)

#어 좀..
def divisor4(x):
    num = []
    y = int(x**(1/2)+1)
    for i in range(1,y):
        if x % i == 0:
            num.append(i)
    l= len(num)
    for j in num[l:0:-1]:
        num.append(int(x / j))
    num.append(x)
    return num

divisor4(1000)

#[문제 58] 표준편차를 구하는 함수 만들기
#stddev(2,3,1,7)
#평균 mean : 관측값들의 합 / 관측값의 수
#편차 = 관측값 - 평균
#편차제곱합 = 편차**2+ ... +편차**2
#분산 = 편차제곱합 / 관측값의 수 (자유)
#표준편차 = math.sqrt(분산)

def sumF(*x):
    total = 0
    for i in x:
        total = total + i
    return total

def mean(*x):
    sum_num = sumF(*x)
    result = sum_num/len(x)
    return result

def variance(*y):
    #가변으로 받았으면 가변으로 전해주어야 한다.
    #가변으로 안 전하면 
    #TypeError: unsupported operand type(s) for +: 'int' and 'tuple'
    #에러가 난다.
    mid = mean(*y)
    dis = 0
    for i in y:
        dis = dis + (i - mid)**2
    result = dis/len(y)
    return result

def variance2(*y):
    total = 0
    for i in y:
        total = total + i
    mid = total/len(y)
    
    dis = 0
    for i in y:
        dis = dis + (i - mid)**2
    result = dis/len(y)
    return result
    
import math 

def stddev(*z):
    var = variance(*z)
    std = math.sqrt(var)
    return std

def stddev2(*z):
    total = 0
    for i in z:
        total = total + i
    mid = total/len(z)
    
    dis = 0
    for i in z:
        dis = dis + (i - mid)**2
    result = dis/len(z)
    var = result
    std = math.sqrt(var)
    return std

mean(1,2,3,4)
variance2(1,2,3,4)
stddev2(1,2,3,4)

variance(1,2,3,4)
stddev(1,2,3,4)

variance(2,2,2,2)
stddev(3,3,3,3)

#라이브러리 만들어 보기 
#이제 위에 만든 함수들을 메모장에 옮겨서 저장하자
#(주석은 제거하고 올린다.)
#파일 확장자는 py로 함

#
import sys
#폴더 확인
sys.path
#path에 폴더 추가 
sys.path.append('C:\\Workspace\\Python_Space\\myPython')
#위의 폴더에 stats.py 파일을 만들고 저장하자
#그 안에 위의 함수들이 있는데 주석은 빼고 넣자  
#라이브러리를 불러오자 
import stats
#stats에서 추가된 함수들을 확인하자 
dir(stats)
stats.sumF(2,3,4,5)

#그리고 cmd에 가서도 해 보자 

#만약 라이브러리 이름을 안 쓰고 싶으면
from stats import mean, variance, stddev, sumF

mean(2,3,4,5,6,7)
variance(3,4,5,6,7)
stddev(4,5,6,7,8,9)

#시스템속성 추가
#시스템변수에 PYTHONPATH, 폴더경로 로 추가하기 
#하면 path에 폴더추가 안 해도 된다.

