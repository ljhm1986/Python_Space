# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:19:55 2019

@author: stu11
"""
#############################################################
#9/9#
#[문제 59] factorial 함수를 생성하세요
#5! = 5*4*3*2*1

def factorial(x):
    fac = 1
    if not isinstance(x, int):
        print('정수가 아닙니다.')
        return 
    if x > 0:
        for i in range(1,x+1):
            fac = fac * i
        print('{}! = {}'.format(i,fac))
    elif x == 0:
        print('{}! = {}'.format(x,fac))
    else:
        print("1보다 더 큰 수를 입력하세요.")
    

factorial(0)


def factorial2():
    x = int(input('0보다 더 큰 정수를 입력하세요 : '))
    fac = 1 
    if x >= 0:
        for i in range(1,x+1):
            fac = fac * i
        print('{}! = {}'.format(i,fac))
    else:
        print("1보다 더 큰 수를 입력하세요.")
        
factorial2()

def factorial3(x):  
    if not isinstance(x, int):
        print('정수가 아닙니다.')
        return  
    if x >= 1:
        y = factorial3(x-1)
        return y * x
    elif x == 0:
        return 1
    else:
        print("1보다 더 큰 수를 입력하세요.")
    
factorial3(2.5)
factorial3(5)
factorial3(4)

#while 문을 이용한 
def factorial4(x):
    fac = 1
    
    if 0 > x:
        return
    else:
        while 1 <= x:#x=0이면 반복문 안 돌아간다.
            fac *= x
            x -= 1
    return fac

factorial4(5)

list(range(5,0,-1))#[5, 4, 3, 2, 1]

def factorial5(x):
    fac = 1
    if x < 0:
        return 
    for i in range(x,0,-1):
        fac = fac * i
    return fac

factorial5(-2)

#factorial 물어보는 이유
#-function, 반복문을 알고있는가, stack 구조를 알고 있는가? 
#쌓을때는 push, 꺼낼때는 맨 위에부터 pop
#list에 쌓을때 append,제거할때 pop을 사용하고 있었다.

#재귀호출
#-자기 자신을 다시 호출하는 기능
#-함수안에서 자신의 함수를 호출하는 기능
#-반복문과 stack구조를 의미한다.

#stack
#- 바닥부터 데이터를 쌓는 개
#- LIFO(Last In First Out) : 마지막으로 들어오는 데이터가 제일 먼저 나간다.
#후힙선출
#-push : 마지막 데이터 위치에 데이터를 입력한다.
#-pop : 마지막 데이터 위치에 데이터를 꺼내는 작업
#-top / peek : 제일 최근에 들어간 데이터 / 가장 위에 있는 데이터 

stack = []
def push(n):
    global stack
    stack.append(n)

def pop():
    global stack
    if len(stack) == 0:
        return None
    return stack.pop()

push(1)
push(2)
push(3)
stack
pop()
pop()
pop()

#제귀호출에서 주의, 멈추는 조건 필요
def cn(n):
    print(n)
    cn(n-1)
cn(5)

#종료조건을 추가하자 
def cn2(n):
    if n <= 0:
        print('종료')
    else:
        print(n)
        cn2(n-1)
cn2(5)

def factorial6(n):
    if n == 1 or n == 0:
        return 1
    return n*factorial6(n-1)

factorial6(5)
#factorial6(5) -> return 5*factorial6(4) push
#factorial6(4) -> return 4*factorial6(3) push
#factorial6(3) -> return 3*factorial6(2) push
#factorial6(2) -> return 2*factorial6(1) push
#factorial6(1) -> return 1 push pop
#이제 값을 구할 수 있다.
#factorial6(2) -> return 2*1 pop
#factorial6(3) -> return 3*2*1 pop
#factorial6(4) -> return 4*3*2*1 pop
#factorial6(5) -> return 5*4*3*2*1 pop

#queue
#-데이터가 입력되면 입력된 순서대로 쌓고 
#먼저 들어온 데이터부터 처리하는 자료구조
#-FIFO(First In First Out) : 처음 들어온 데이터가 처음 먼저 나간다.
#선입선출

queue = []
queue.append('a')
queue.append('b')
queue.append('c')

queue[0]
#그리고 무조건 맨 앞에 것만 뽑아낸다.
queue.pop(0)
queue.pop(0)
queue.pop(0)

#최대공약수
#18  1 2 3 6 9 18
#24  1 2 3 4 6 8 12 24
#gcd를 구해보자 
#1. 큰수를 작은수로 나눈다. 큰수%작은수 = 나머지
#나머지가 없이 딱 나누어지면 작은수가 최대 공약수가 된다.
#24 % 18 = 6
#     18 % 6 = 0
#100 % 28 = 16
#      28 % 16 = 12
#           16 % 12 = 4
#                12 % 4 = 0

def gcd(x,y):
    if x < y:
        x, y = y, x
    
    z = x % y
    
    if z == 0:
        return y
    else:
        return gcd(y,z)

gcd(100,28)
gcd(43,54)
k = gcd(76,33)
gcd(5,7)

def gcd2(x,y):
    if x < y:
        x, y = y, x
       
    while y > 0:
        remainder = x % y
        if remainder == 0:
            return y
        else:
            x = y
            y = remainder

gcd2(100,28)
gcd2(4000,4)

def gcd3(x,y):
    if x < y:
        x, y = y, x
       
    while y != 0:
        remainder = x % y
        x = y
        y = remainder
    
    return x

gcd3(6,8)
gcd3(33,55)

#유클리드 알고리즘
#1. 두 수 x,y(x>y)입력으로 들어온다.
#2. y=0이면 x를 출력하고 종료한다.
#3. x를 y로 나누어 떨어지면 y를 출력하고 알고리즘 종료
#4. 그렇지 않으면 x를 y로 나눈 나머지가 있으면 y를 x에 대입, 나머지는 y에 대입
#->3. 반복 수행 

#패키지 - 관련성있는 서브프로그램을 모아서 관리
#class 
#날짜
import datetime
#datetime안에 무엇이 있는지 확인하자
dir(datetime)
dir(datetime.date)
dir(datetime.datetime)
dir(datetime.datetime.now())
dir(datetime.datetime.today())
#library.class.function 식으로 불러온다.
datetime.date.today()#datetime.date(2019, 9, 9)
datetime.datetime.now()#datetime.datetime(2019, 9, 9, 14, 0, 54, 65270)
datetime.datetime.today().year
datetime.datetime.today().month
datetime.datetime.today().day
d = datetime.date.today()
d.day
type(d)#datetime.date

datetime.datetime.now().year
datetime.datetime.now().month
datetime.datetime.now().day
datetime.datetime.now().hour
datetime.datetime.now().minute
datetime.datetime.now().second
datetime.datetime.now().microsecond
datetime.datetime.now().date()#datetime.date(2019, 9, 9)
datetime.datetime.now().time()#datetime.time(17, 48, 8, 386331)
datetime.datetime.now().weekday()#0

#[문제 60] 오늘 무슨 요일인지 출력해주세요
datetime.datetime.now().weekday()#0
weekdays = ['월요일','화요일','수요일','목요일','금요일','토요일','일요일']
weekdays[datetime.datetime.now().weekday()]


d = datetime.datetime.now()
type(d)
type(d.year)
print(d)

d.strftime('%x')#현재날짜 월/일/년도
d.strftime('%X')#현재 시간 시:분:조
d.strftime('%Y')#년도 4자리
d.strftime('%y')#년도 뒤 2자리
d.strftime('%m')#달
d.strftime('%d')#일
d.strftime('%B')#달(영어)
d.strftime('%H')#시간 24
d.strftime('%I')#시간 12
d.strftime('%M')#분
d.strftime('%S')#초
d.strftime('%A')#요일(영문)
d.strftime('%a')#요일약어
d.strftime('%c')#요일 달 일 시분초 년도 
d.strftime('%p')#AM, PM
d.strftime('%j')#누적일
d.strftime('%U')#누적주(일요일 시작)
d.strftime('%W')#누적주(월요일 시작)
d.strftime('%w')#요일
d.strftime('%z')#timezone

#strftime : date -> char
#strftime : char -> date

datetime.datetime.strptime('2019-09-09 14:24:10','%Y-%m-%d %H:%M:%S') 

d = datetime.date(2019,9,9)
t = datetime.time(14,28,20)
datetime.datetime.combine(d,t)
#datetime.datetime(2019, 9, 9, 14, 28, 20)

#날짜 - 날짜
datetime.datetime(2019,12,23)-datetime.datetime(2019,6,18)
#datetime.timedelta(days=188)
(datetime.datetime(2019,12,23)-datetime.datetime(2019,6,18)).days
#188

#날짜 + 날
datetime.datetime(2019,9,9) +datetime.timedelta(days=105)
#datetime.datetime(2019, 12, 23, 0, 0)
datetime.datetime(2019,9,9) +datetime.timedelta(hours=105)
datetime.datetime(2019,9,9) +datetime.timedelta(minutes=105)
datetime.datetime(2019,9,9) +datetime.timedelta(seconds=105)

#[문제 61] 함수에 인수값으로 현재날짜, 일수 정보를 입력하면 더한
#날짜정보가 리턴하는 next_day 함수를 생성하세요

def next_day(y,m,d,n):
    result = datetime.date(y,m,d) + datetime.timedelta(days = n)
    return result

next_day(2019,9,9,12)


def next_day2(ti,n):
    date = ti + datetime.timedelta(days = n)
    return date

next_day2(datetime.date.today(),15)
next_day2(datetime.date(2019,9,8),15)

def next_day3(today,n):
    date = datetime.datetime.strptime(today,'%Y-%m-%d')
    date = date + datetime.timedelta(days = n)
    return date

next_day3('1999-5-6',7)

dir(datetime.timedelta)
start = datetime.datetime.now()
end = datetime.datetime.now()
(end - start).seconds
(end - start).microseconds
(end - start).total_seconds()
(end - start).days

#[문제 62] 아래와 같은 결과가 출력될수 있도록
#프로그램을 생성하세요

def sumF():
    print('1에서 천만까지 짝수합, 홀수합 구합니다.')
    start = datetime.datetime.now()
    sum_odd = 0
    sum_even = 0
    for i in range(1,10000001):
        if i % 2 == 0:
            sum_even += i
        else:
            sum_odd += i
            
    end = datetime.datetime.now()
    print('1에서 천만까지 짝수합 {}'.format(sum_even))
    print('1에서 천만까지 홀수합 {}'.format(sum_odd))
    time = end - start
    total_seconds = time.total_seconds()
    print('처리시간 :00:00:{}'.format(total_seconds))
    print('처리시간 : ', time)
    print('처리시간 millisecond(1/1000) : {}ms'.format(int(total_seconds*1000)))
    
sumF()

import time        
dir(time)
#1970년 1월 1일 0시 0분 0초를 기준으로 지난 시간을 초단위로 리턴
time.time()

time.localtime()
#time.struct_time(tm_year=2019, tm_mon=9, tm_mday=9, tm_hour=18,
#tm_min=24, tm_sec=37, tm_wday=0, tm_yday=252, tm_isdst=0)
time.localtime().tm_year
time.localtime().tm_mon
time.localtime().tm_mday
time.localtime().tm_hour
time.localtime().tm_wday#요일
time.localtime().tm_yday#누적일
time.localtime().tm_isdst#서머타임일 경우 1, 아닐경우 0, 모름 -1
time.strftime('%Y %z', time.localtime())#년도와 timezone만 보기 
time.strftime('%H', time.localtime())#시간만 보기

for i in range(10):
    print(i)
    time.sleep(2)#2초 있다가 하기 
    
import calendar

#달력을 보기 
#올해 달력을 보자 
print(calendar.calendar(2019))
calendar.prcal(2019)
#9월 달력을 보자 
calendar.prmonth(2019,9)
calendar.weekday(2019,9,9)#숫자 요일 보기 
'월화수목금토일'[calendar.weekday(2019,9,9)]
'월화수목금토일'[time.localtime().tm_wday]

calendar.monthrange(2019,9)#그 달의 첫째날짜의 요일과 마지막일

#파일 입출력
#파일 생성
#파일 객체 = open('C:\WorkSpace\Python_Space\data\test.txt',mode)
#mode
#r : 읽기
#w : 쓰기(파일안에 원본 데이터는 지우고 작성된다.)
#a : 추가 

f = open('C:\\WorkSpace\\Python_Space\\data\\test.txt','w')
for i in range(1,11):
    txt = "%d 오늘 하루도 행복하자\n"%i
    f.write(txt)  
f.close()

#다시 파일을 열어서 입력해보자 
f = open('C:\\WorkSpace\\Python_Space\\data\\test.txt','w')
for i in range(20,31):
    txt = "%d 오늘 하루도 행복하자\n"%i
    f.write(txt)    
f.close() 
#그럼 내용이 바뀌어 있다. 
#내용을 추가를 해 보자    
f = open('C:\\WorkSpace\\Python_Space\\data\\test.txt','a')
for i in range(31,41):
    txt = "%d 오늘 하루도 행복하자\n"%i
    f.write(txt)
f.close()

#파일 읽기를 해 보자 
f = open('C:\\WorkSpace\\Python_Space\\data\\test.txt','r')
data = f.readline()
data#1줄만 있다.
print(data)
print(f.readline())#다음줄을 볼 수 있다.
f.close()

#반복해서 출력해 보자 
f = open('C:\\WorkSpace\\Python_Space\\data\\test.txt','r')
while True:
    data = f.readline()
    if not data:
        break
    print(data, end = '')
f.close()

###############################################################
#9/10#

import os
#해당 파일이 존재하는지 확인하자 
os.path.exists("C:\\WorkSpace\\Python_Space\\data\\test.txt")
#True

#읽기 전용으로 불러들이자 
file = open('C:\\WorkSpace\\Python_Space\\data\\test.txt','r')
while True:
    data = file.readline()
    if not data:
        break
    print(data, end = '')
file.close()

file = open('C:\\WorkSpace\\Python_Space\\data\\test.txt','r')
data = file.readlines()
data
print(data)
file.close()
data

#file을 open, close를 동시에 하면 어떨까?
#pl/sql을 할때 cousor(sql문 실행 메모리 영역) 를 활용할때 처럼
#(커서선언 -> 커서 open -> fetch -> 커서 close)
#이때 fetch 문을 for 문으로 바꾸어서 동시에 했다.
#with문 : 파일객체를 자동으로 열고 닫아주는 문 

with open('C:\\WorkSpace\\Python_Space\\data\\test.txt','w') as file:
    for i in range(1,11):
        txt = '%d 오늘 하루도 행복하자\n'%i
        file.write(txt)
        
txt = ['야!! 추석이다','오늘 하루도 신난다. 크아아악']

with open('C:\\WorkSpace\\Python_Space\\data\\test1.txt','w') as file:
    for i in txt:
        file.write(i+'\n')

with open('C:\\WorkSpace\\Python_Space\\data\\test1.txt','r') as file:
    data = file.readlines()
    for i in data:
        print(i, end='')

##################################        
#이제 sql developer 로 가서 익스포트를 하자
#oracle sql developer - 도구 - 데이터베이스 익스포트

#DDL익스포트는 안 함 
#파일 형식 - csv 형식
#헤더 - column name 까지 포함 
#행 터미네이터 - 환경 기본값  
#둘러싸기 - 없음 (문자는 "가 둘러싼 형태로 있는데 없게 하자)
#인코딩 - MS949

#토글 - 테이블 만 

#조회 - 익스포트할 테이블만
###################################
#csv 파일을 불러보자 
import csv
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
emp_csv#<_csv.reader at 0x791c388>

for i in emp_csv:
    print(i[2],i[7])
  
#내용물을 보고 싶으면 다음과 같이 list로 옮겨놓자
emp_see = []
for i in emp_csv:
    emp_see.append(i)
emp_see
emp_see[0]

#column중에 하나를 key값으로 하고 dictionary에 옮겨놓자 
#key는 column의 값이 unique한 것으로 고르자  
emp_see2 = {}
for i in emp_csv:
    emp_see2[i[0]] = i[1:]
emp_see2['100']
emp_see2.keys()
emp_see2.values()
emp_see2.items()

file.close()

#column이름은 보기 싫으면
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
#첫번째 줄을 읽으면서 pop한거처럼 된다.
next(emp_csv)
next(emp_csv)

for i in emp_csv:
    print(i[2],i[7])
    
file.close()

#[문제 63] emp.csv 파일에서 last_name, last_name 길이를 출력해주세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
#next()를 사용해서 last_name이 몇번째 column인지 보자 
next(emp_csv)#3번째이다

for i in emp_csv:
    print(i[2], len(i[2]))
#뭐지 한 번하고 또 하면 안 나오네 
    
file.close()

#[문제 64] employee_id, last_name, salary * 12 값을 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#1번째, 3번째, 8번째
for i in emp_csv:
    print(i[0], i[2], float(i[7])*12)
file.close()

#[문제 65] last_name, commission_pct 를 출력하는데 
#commission_pct값이 ''이면 0으로 출력해주세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)

for i in emp_csv:   
    print(i[2], ifnull(i[8],0))
#값이 없으면 아무것도 나오지 않는다.
    
def ifnull(arg1, arg2):
    if arg1 == '':
        return arg2
    return arg1

file.close()
#
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)

for i in emp_csv:   
    print(i[2], 0 if i[8] == '' else i[8])
    

file.close()


#[문제 66번] last_name은 대문자, job_id는 소문자로 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3번쨰, 7번째

for i in emp_csv:   
    print(i[2].upper(), i[6].lower())
    
file.close()

#[문제 67번] last_name은 첫글자만 추출해서 소문자로 출력해주세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3번째

for i in emp_csv:   
    print(i[2][0].lower())
    
file.close()


#[문제 68번] last_name의 두번쨰부터 마지막까지 추출해서 대문자로 출력해주세
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3번째

for i in emp_csv:   
    print(i[2][1:].upper())
    
file.close()

#[문제 69] 이름을 입력하면 첫글자는 대문자, 나머지는 소문자를 
#추력하는 initcap 함수를 이용해서 이름을 출럭하기
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3번째

def initcap(arg):
    return arg[0].upper() + arg[1:].lower()

for i in emp_csv:   
    print(initcap(i[2]))
    
file.close()

#[문제 70] 이름을 입력하면 제일 뒤에 있는 철자는 대문자, 
#앞의 문자는 소문자를 출력하는 tailcap함수를 생성하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3번째

def tailcap(arg):
    return arg[0:-1].lower() + arg[-1].upper()

for i in emp_csv:   
    print(tailcap(i[2]))
    
file.close()

#[문제 71] last_name, salary를 출력하는데 salary값 중에
#0대신 *를 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3번째, 8번째

for i in emp_csv:   
    print(i[2], i[7].replace('0','*'))
    
file.close()

#[문제 72] last_name, salary*12 + commission_pct값을 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3번째, 8번째, 9번째

def ifnull(arg1, arg2):
    if arg1 == '':
        return arg2
    return arg1

for i in emp_csv:   
    print(i[2], float(i[7])*12+float(ifnull(i[8],0)))
    
file.close()

#[문제 73] last_name, hire_date 는 요일(한글)을 출력하세요.
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3, 6

import datetime
weekdays = ['월요일','화요일','수요일','목요일','금요일','토요일','일요일']
week_num = datetime.datetime.strptime(i[5],'%Y-%m-%d').weekday()

for i in emp_csv:
    print(i[2],weekdays[week_num])

file.close()

#[문제 74] last_name, 근무한 일수를 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)

import datetime
end = datetime.datetime.now()
hire_day = datetime.datetime.strptime(i[5],'%Y-%m-%d')


for i in emp_csv:
    print(i[2],(end-hire_day).days)

file.close()

#다음과 같을때
end = datetime.datetime.now()
hire_day = datetime.datetime.strptime('2019-01-01','%Y-%m-%d')
(end-hire_day).days#252

datetime.date.today() - hire_day#이건 안 된다. 모양이 맞아야 한다.
datetime.date.today() - hire_day.date()#이렇게 해야 된다.
(datetime.date.today() - hire_day.date()).days

#[문제 75] 오늘부터 이번달 말일까지 몇일 남았는지 출력하세요
import datetime
import calendar

today = datetime.datetime.now()
last_day = calendar.monthrange(2019,9)[1]
remainderDay = (datetime.datetime(2019,9,last_day) - today).days
print(remainderDay)

##
from datetime import date
from calendar import monthrange

monthrange(2019,9)[1] - date.today().day

#[문제 76] 사원번호가 100번 사원의 last_name, salary를 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3, 8

for i in emp_csv:
    if int(i[0]) == 100:
        print(i[2],i[7])

file.close()

#[문제 77] 급여가 10000 이상인 사원들의 last_name, salayr를 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3, 8

for i in emp_csv:
    if int(i[7]) >= 10000:
        print(i[2], i[7])

file.close()

#[문제 78] 2001-01-13일에 입사한 사원의 last_name, hire_date를 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3, 6

for i in emp_csv:
    if i[5] == '2001-01-13':
        print(i[2],i[5])

file.close()

#time library를 사용한다면
import time  

for i in emp_csv:
    if (time.strptime(i[5],'%Y-%m-%d') == 
        time.strptime('2001-01-13','%Y-%m-%d')):
        print(i[2],i[5])

#[문제 79] 2002년도에 입사한 사원들의 last_name, hire_date를 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)

for i in emp_csv:
    if i[5][0:4] == '2002':
        print(i[2],i[5])
        
file.close()

#time library를 사용한다면
import time  

for i in emp_csv:
    if (time.strptime(i[5],'%Y-%m-%d').tm_year == 
        time.strptime('2002','%Y').tm_year):
        print(i[2],i[5])
        
#[문제 80] 9월에 입사한 사원들의 last_name, hire_date를 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)      

for i in emp_csv:
    if i[5][5:7] == '09':
        print(i[2],i[5])

file.close()

#time library를 사용한다면
import time  

for i in emp_csv:
    if(time.strptime(i[5],'%Y-%m-%d').tm_mon == 
       time.strptime('09','%m').tm_mon):
        print(i[2],i[5])
        
#[문제 81] job이 ST_CLERK이고 급여가 3000 이상인 사원들의 
#last_name, job_id, salary를 출력하세요.
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3,7,8
    
all_num = 0
part_num = 0

for i in emp_csv:
    all_num += 1
    if (i[6] == 'ST_CLERK') & (int(i[7]) >= 3000):
        part_num += 1
        print(i[2],i[6],int(i[7]))
print('all_num : {}, part_num : {}'.format(all_num, part_num))
print('비율 {}%'.format(round((part_num*100)/all_num)))

file.close()
        
#[문제 82] 급여가 2500~3500 사이인 사원들의 
#last_name, salary를 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3,7,8
   
all_num = 0
part_num = 0
 
for i in emp_csv:
    all_num += 1
    if (int(i[7]) >= 2500) & (int(i[7]) <= 3500):
        part_num += 1
        print(i[2],int(i[7]))
print('all_num : {}, part_num : {}'.format(all_num, part_num))
print('비율 {}%'.format(round((part_num*100)/all_num)))

file.close() 
   
#[문제 83] job이 AD_VP, AD_PRES인 사원들의 
#last_name, job_id를 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3,7,8

all_num = 0
part_num = 0
    
for i in emp_csv:
    all_num += 1
    if i[6] in ['AD_VP', 'AD_PRES']:
        part_num += 1
        print(i[2],i[6])
print('all_num : {}, part_num : {}'.format(all_num, part_num))
print('비율 {}%'.format(round((part_num*100)/all_num)))

file.close()     

#[문제 84] job이 AD_VP, AD_PRES가 아닌 사원들의 
#last_name, job_id를 출력하세요
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
next(emp_csv)#3,7,8

all_num = 0
part_num = 0
    
for i in emp_csv:
    all_num += 1
    #in 앞에 not만 추가하면 된다.
    if i[6] not in ['AD_VP', 'AD_PRES']:
        part_num += 1
        print(i[2],i[6])
print('all_num : {}, part_num : {}'.format(all_num, part_num))
print('비율 {}%'.format(round((part_num*100)/all_num)))

file.close()

#[문제 85] 커미션이 null 인 사원의 이름, 급여, 커미션을 출력하세요
with open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r') as file:
    emp_csv = csv.reader(file)
    print(next(emp_csv))
    
    for i in emp_csv:
        if i[8] == '':
            print(i[2],i[7],i[8])

#[문제 86] 커미션이 null 이 아닌 사원들의 이름, 급여, 커미션을 출력하세요
with open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r') as file:
    emp_csv = csv.reader(file)
    print(next(emp_csv))
    
    for i in emp_csv:
        if i[8] != '':
            print(i[2],i[7],i[8])            
            

#[문제 87] last_name의 첫글자가 S로 시작하는 사원들의 이름과 급여를 출력하세요
with open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r') as file:
    emp_csv = csv.reader(file)
    print(next(emp_csv))
    
    for i in emp_csv:
        if i[2].startswith('S'):
            print(i[2],i[7])              

#[문제 88] last_name의 두번째 철자가 i인 사원들의 이름과 월급을 출력하세요
with open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r') as file:
    emp_csv = csv.reader(file)
    print(next(emp_csv))
    
    for i in emp_csv:
        if i[2][1]=='i':
            print(i[2],i[7])

#[문제 89] 50번 부서 사원들이 last_name, salary를 출력하는데 
#last_name을 오름차순으로 정렬하세
with open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r') as file:
    emp_csv = csv.reader(file)
    print(next(emp_csv))
    
    dic = {}
    
    for i in emp_csv:
        if i[10] == '50':
            dic[i[2]] = i[7]
    for key, value in sorted(dic.items()):
        print(key, value)
    
    #salary 기준으로 정렬하려면 
    import operator
    for key, value in sorted(dic.items(),key = operator.itemgetter(1)):
        print(key, value)

##dictionary가 아니라 list를 이용해서 풀어보면 
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
print(next(emp_csv))
emp_50 = []

for i in emp_csv:
    if i[-1] == '50':
        emp_50.append(i)
    
file.close()
emp_50#중첩 list모양이 된다.
print(emp_50)
for i in emp_50:
    print(i)

import operator
emp_50_sorted = sorted(emp_50, key = operator.itemgetter(7))

for i in emp_50_sorted:
    print(i[2],i[7])

#또는 처음에 2개의 column만 골라내서 emp_50을 만들자 
file = open('C:\\WorkSpace\\Python_Space\\data\\emp.csv','r')
emp_csv = csv.reader(file)
print(next(emp_csv))
emp_50 = []

for i in emp_csv:
    if i[-1] == '50':
        emp_50.append([i[2],i[7]])
    
file.close()
emp_50#중첩 list모양이 된다.

import operator
emp_50_sorted = sorted(emp_50, key = operator.itemgetter(1))

for i in emp_50_sorted:
    print(i[0],i[1])

#############################################################
#9/11#
''' pickle: 변수 형태를 그대로 유지해서 파일에 저장하고 불러올 수 있는 모듈
바이너리 형태로 저장된다. '''
import pickle
lst = ['a', 'b', 'c']
file = open('C:\\WorkSpace\\PythonSpace\\Python_Space\\data\\lst.txt', 'wb') # wb: 바이너리 형식으로 쓰기
pickle.dump(lst, file)
file.close()

file = open('\\WorkSpace\\PythonSpace\\Python_Space\\data\\lst.txt', 'rb') #rb: 바이너리 파일 읽기
lst_temp = pickle.load(file)
file.close()
lst_temp



''' lambda 함수
- 이름 없는 한줄짜리 함수
- 가독성이 좋다.
- 성능도 좋다. '''
def f1(x,y):
    return x*y
f1(2,3)

(lambda x, y : x*y)(2,3)

(lambda x,y,z : x+y+z)(4,5,6)

#이건 안 되네 
(lambda *arg :
total = 0 
for i in arg: 
    total += i 
total)(4,5,6,7)

(lambda *arg : arg)(4,5,6,7,8,9)

#이제 
import collections
#Counter() : dictionary의 일종, 원소는 key로 원소의 갯수는 value에 저장한다.
collections.Counter(['A', 'B', 'AB', 'O', 'A', 'B', 'AB', 'O'])
#Counter({'A': 2, 'B': 2, 'AB': 2, 'O': 2}) 각 원소의 갯수를 보여준다. 

cnt = collections.Counter()
type(cnt)
cnt.update('aaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbcccccccgggggzzzzzzzzzzzzxx')
print(cnt)#Counter({'a': 20, 'b': 17, 'z': 12, 'c': 7, 'g': 5, 'x': 2})
cnt.update({'c': 3, 'y': 10})
print(cnt)#Counter({'a': 20, 'b': 17, 'z': 12, 'c': 10, 'y': 10, 'g': 5, 'x': 2})
cnt['a']
for k, v in cnt.items():
    print(k, v)

cnt1 = collections.Counter(['a', 'b', 'c', 'd', 'b', 'a'])
cnt2 = collections.Counter('aeroplane')
cnt1 + cnt2
cnt1 - cnt2
cnt1 & cnt2 # 교집합
cnt1 | cnt2 # 합집합, +와 결과가 같게 나왔다. 




''' exception: 실행 중에 발생한 오류

1. SyntaxError: 문법 오류
x := 1
print x
print('happy)

2. NameError: 참조하는 변수가 없을 경우
x = 1
y = 2
print(z)

3. IndexError: 인덱스 범위를 틀리게 입력할 경우
x = [1,2,3]
x[3]
dic = {'name':'홍길동', 'addr':'서울시'}
dic['age'] #KeyError : 'age'

4. ZeroDivisionError: 0으로 나누는 경우
10/0

5. AttributeError: 모듈, 클래스에 있는 잘못된 속성을 사용하려는 경우
import time
time.time()
time.month() #AttributeError: module 'time' has no attribute 'month'

6. ValueError: 참조값이 없는 경우
x = [10,20,30]
x.remove(50)
x.index(50)

7. FileNotFoundError: 없는 파일을 읽어들이는 경우
file = open('test.txt', 'r')

8. TypeError: 자료형이 틀린 경우
x = [1,2]
y = (1,2)
x + y

'''

def divide(x,y):
    print('x: {}'.format(x))
    print('y: {}'.format(y))
    return x/y
try:
    z = divide(10,0)
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.")
except:
    print("오류가 발생했습니다.")
else:
    print("결과 {}".format(z))
finally: # 오류의 발생과 관계없이 무조건 마지막에 수행됨
    print("프로그램 종료") 

#이거 안 되는데 
def sum(*x, **y):
    print('x : {}'.format(*x))
    print('y : {}'.format(**y))
    z = 0
    for i in x:
        for j in y:
            z = z + 1
    return z

def errerTest(functionName, number1,number2):
    result = 0.0
    print('number1 : {}'.format(number1))
    print('number2 : {}'.format(number2))
    try:
        if functionName == 'divide':
            result = divide(number1,number2)
        if functionName == 'sum':
            result = sum(number1,number2)
    except ZeroDivisionError:
        print('ZeroDivisionError')   
    except RuntimeError:
        print('RuntimeError')
    except:
        print('errer')
    else:
        print(result)
    finally:
        print('exit')

sum(6,7)
divide(8,2)
errerTest('divide',8,0)
errerTest('sum',6,7)
    
''' raise: 사용자가 발생시키는 오류 '''
def func(arg):
    try:
        if arg < 1 or arg > 10:
            raise Exception("유효하지 않은 숫자입니다. {}".format(arg))
        else:
            print("입력한 숫자는 {}입니다.".format(arg))
    except Exception as err:
        print("오류가 발생했습니다. {}".format(err))
        
func(100)

#문제

''' [문제90] 부서별 급여의 총액을 구하시고 부서별로 오름차순 정렬하세요. '''
import csv
file = open("C:\\WorkSpace\\PythonSpace\\Python_Space\\data\\emp.csv", "r")
emp_csv = csv.reader(file)
header = next(emp_csv)
header

#보면 모두 문자열로 저장되어 있다. 
for i in emp_csv:
    print(i)

dept_sum = {} # key: 부서, value: 급여의 합
for i in emp_csv:
    if i[-1] in dept_sum.keys():
        dept_sum[i[10] if i[10] != '' else '999'] = int(dept_sum[i[10]]) + int(i[7])
    else:
        dept_sum[i[10] if i[10] != '' else '999'] = int(i[7])
# 부서번호가 없는 사원의 부서번호는 999로
# 빈 문자열은 int로 변환이 안됨. -> 빈문자열을 먼저 제거하고 하면 될듯?
for k, v in dept_sum.items():
    print(k, v)

# 나의 정렬 방법, list 사용
#1. dept_sum의 key값만 뽑아내 list로 만들어서 sorted한다.
dept_id = []
for i in range(len(dept_sum)):   
    dept_id.append(int(list(dept_sum.keys())[i]))
dept_id = sorted(dept_id)
dept_id #[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 999]

#2. list값에 맞는 값들을 순서대로 출력한다.
for i in dept_id:
    print(i, dept_sum[str(i)])

# 선생님의 정렬 방법, dictionary 사용
#1. 새로운 dictionary를 만든다.
new = {}
for k, v in dept_sum.items():
    new[int(k)] = v

#2. dictionary를 key값 기준으로 정렬한다.
from operator import itemgetter
new_sort = sorted(new.items(), reverse = False, key=itemgetter(0))
for k, v in new_sort:
    print(' ' if k == 999 else k, v)
         
''' [문제91] 단어, 알파벳을 입력값으로 넣어서 단어 안의 알파벳 수를 출력하세요. '''
def wordF(word, alpha):
    cn = 0
    for i in range(len(word)):
        if word[i] == alpha:
            cn += 1
    return cn
wordF('happy', 'p')
wordF('happy', 'e')

#collections를 이용해서 풀어보자 
import collections
def wordF2(word, alpha):
    cnt = collections.Counter()
    cnt.update(word)
    return cnt[alpha]

wordF2('abcdefg','a')
wordF2('happy', 'p')

''' [문제92] 단어를 입력값으로 넣어서 알파벳을 출력하는데 중복되는 알파벳은 하나만
출력하세요. '''
def alphaF(word):
    alphaset = set() # list여도 될듯
    for i in word:
        if i in alphaset:
            continue
        else:
            alphaset.add(i)
            print(i, end = ' ')
alphaF('happy')
alphaF('happy new year')


def alphaF2(word):
    alphaset2 = list()
    for i in word:
        if i in alphaset2:
            continue
        else:
            alphaset2.append(i)
            print(i, end = ' ')
    print('\n')
    print(alphaset2)

alphaF2('happy new year')

''' [문제93] 철자의 빈도수를 출력하세요.
alphaF('intelligence')
{'i': 2, 'n': 2, 't': 1, 'e': 3, 'l': 2, 'g': 1, 'c': 1} '''
def alphaF3(word):
    alphaset = dict()
    for i in word:
        if i in alphaset:
            alphaset[i] = int(alphaset[i]) + 1
        else:
            alphaset[i] = 1
    return alphaset
alphaF3('intelligence')



''' [문제94] 부서별 인원수를 출력해주세요. '''
import csv
import operator
file = open("C:\\WorkSpace\\PythonSpace\\Python_Space\\data\\emp.csv", "r")
emp_csv = csv.reader(file)
header = next(emp_csv)
header
dept_cn = {} # key: 부서, value: 인원수
for i in emp_csv:
    if i[10] == '':
        i[10] = 999
    else:
        i[10] = int(i[10])
        
    if i[10] in dept_cn.keys():
        dept_cn[i[10]] += 1 
    else:
        dept_cn[i[10]] = 1
dept_cn
dept_sort = sorted(dept_cn.items(), reverse = False, key=operator.itemgetter(0))
for k, v in dept_sort:
    print('-' if k == 999 else k, v)
file.close()

# lambda 사용한 선생님의 답
file = open("C:\\WorkSpace\\PythonSpace\\Python_Space\\data\\emp.csv", "r")
emp_csv = csv.reader(file)
header = next(emp_csv)
header
dept_cnt = {}
for emp_list in emp_csv:
    x = (lambda arg: int(arg) if arg != '' else 999)(emp_list[10])
    if x in dept_cnt.keys():
        dept_cnt[x] = dept_cnt[x] + 1
    else:
        dept_cnt[x] = 1
dept_cnt_sorted = sorted(dept_cnt.items(), reverse=False, key=operator.itemgetter(0))
for k,v in dept_cnt_sorted:
        print('--' if k == 999 else k,v)
file.close()



''' [문제95] 숫자를 입력값으로 받은 후 숫자가 짝수인지 홀수인지 출력한 후에
그 숫자값을 기준으로 짝수면 짝수형식으로 증값을 10개 출력,
홀수면 홀수형식으로 10개 출력합니다.
만약에 숫자가 들어 오지 않으면 예외사항처리를 해주세요. '''
try:
    arg = input("숫자를 입력해주세요 : ")
    if not str(arg).isnumeric():
        raise Exception("숫자를 입력해주세요.")
    else:
        for i in range(1,11):
            print(int(arg) + 2*i, end = ' ')
except Exception as err:
    print("오류가 발생했습니다. {}".format(err))



# 선생님의 답
try:
    num = int(input("숫자를 입력해주세요 :"))
    if num % 2 == 0:
        print("짝수")
    else:
        print("홀수")
    count = 1
    while count <= 10:
        print(num)
        num += 2
        count += 1
except ValueError as error:
    print(error)
    print("숫자를 입력해주세요")