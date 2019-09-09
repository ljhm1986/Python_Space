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

