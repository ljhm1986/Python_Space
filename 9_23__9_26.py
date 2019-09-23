# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:14:18 2019

@author: stu11
"""

#9/23#
#현재날짜를 보려면 
import datetime

dir(datetime)
dir(datetime.date)
datetime.sys

datet = datetime.datetime.now()
datet
datet.year
datet.month

#pandas 에서 제공하는 함수로 날짜를 보자 
from pandas import Series, DataFrame
import pandas as pd

pd.datetime.now() #datetime.datetime(2019, 9, 23, 18, 16, 34, 624421)
print(pd.datetime.now()) #2019-09-23 18:16:50.617336
pd.datetime.now().year
pd.datetime.now().month
pd.datetime.now().day
pd.datetime.now().hour
pd.datetime.now().minute
pd.datetime.now().second
pd.datetime.now().microsecond

pd.Timestamp.now()
times = pd.Timestamp(year = 2019, month = 9, day = 23, hour = 9,
             minute = 50, second = 10, microsecond = 1, tz = 'Asia/Seoul')
times
print(times)
times.today()#만든 시점의 시간이 아니라 현재시간으로 나온다. 

#문자형 날짜를 날짜형으로 바꾸기 
pd.to_datetime('2019-09-23')
pd.to_datetime('2019-09-23 09:30')
pd.to_datetime('20190923')
pd.to_datetime('09232019')#error
pd.to_datetime('09232019', format = '%m%d%Y')
pd.to_datetime('2019-09-23 09:59:20', format = '%Y-%m-%d %I:%M:%S')
pd.to_datetime('2019-09-23 15:59:20', format = '%Y-%m-%d %H:%M:%S')#%H : 24시간

#DataFrame을 만들어서 날짜형으로 바꾸자,  key 값들이 날짜형과 일치해야 한다.
df = DataFrame({'year':[2018,2019], 'month':[5,12], 'day':[5,23]})
pd.to_datetime(df)

date_str = ["2019, 9, 1","2019, 10, 1"]#,뒤에 공백문자 필요함 
pd.to_datetime(date_str)

#pd.date_range : 시작일과 종료일 또는 시작일과 기간을 입력하면
#범위내의 인덱스를 생성해준다.

pd.date_range("2019-9-1","2019-9-30")
pd.date_range(start = '2019-9-1', periods = 20)#20개 생성함, 차이는 하루  
#시계열쪽에서 사용될 기능이다. 

pd.date_range(start = '2019-9-1',periods=25,freq='H')#25개를 생성, 차이는 1시간 
pd.date_range(start = '2019-9-1',periods=25,freq='3H')#25개를 생성, 차이는 3시간 
pd.date_range(start = '2019-9-1',periods=50,freq='T')#50개를 생성, 차이는 1분 
pd.date_range(start = '2019-9-1',periods=20,freq='4T')#20개를 생성, 차이는 4분 
pd.date_range('2019-9-1',periods=20,freq='S')#20개를 생성, 차이는 1초 
pd.date_range('2019-9-1','2019-9-30',freq = 'B')#평일만 출력
pd.date_range('2019-9-1','2019-9-30',freq = 'w')#일요일만 출력
pd.date_range('2019-9-1','2019-9-30',freq = 'W-MON')#월요일만 출력
pd.date_range('2019-9-1','2019-9-30',freq = 'W-TUE')#화요일만 출력 
pd.date_range('2019-1-1','2019-12-31',freq = 'M')#매달 마지막 날짜
pd.date_range('2019-1-1','2019-12-31',freq = 'MS')#매달 처음 날짜 
pd.date_range('2019-1-1','2019-12-31',freq = 'BM')#매달 마지막 평일
pd.date_range('2019-1-1','2019-12-31',freq = 'BMS')#매달 첫 평일 
pd.date_range('2019-1-1','2019-12-31',freq = 'WOM-1MON')#매달 첫 월요일 날짜
pd.date_range('2019-1-1','2019-12-31',freq = 'WOM-1MON')#매달 두번째 금요일 날짜
pd.date_range('2019-1-1','2019-12-31',freq = 'Q')#매 분기 마지막 날짜
pd.date_range('2019-1-1','2019-12-31',freq = 'Q-JAN')#매 분기 첫달 마지막 날짜
pd.date_range('2019-1-1','2019-12-31',freq = 'Q-DEC')#매 분기 마지막달 마지막 날짜 

#날짜 형식 계산
pd.datetime.now() + 1 # error
pd.datetime.now() + pd.Timedelta('1 days')#현재 시각에 하루 더하기
pd.datetime.now() + pd.Timedelta('8 hours') #현재 시각에 8시간 더하기 
pd.datetime.now() + pd.Timedelta('60 minute')#현재 시각에 60분 더하기
pd.datetime.now() + pd.Timedelta('3600 second')#현재 시각에 3600초 더하기

pd.datetime.now() + pd.DateOffset(years = 1)#1년 뒤
pd.datetime.now() + pd.DateOffset(year = 1)#년도가 1년으로 바뀐다.
pd.datetime.now() + pd.DateOffset(months = 1)#1달 뒤
pd.datetime.now() + pd.DateOffset(days = 1)#하루 뒤
pd.datetime.now() + pd.DateOffset(hours = 1)#한 시간 뒤
pd.datetime.now() + pd.DateOffset(minutes = 100)#100분 뒤
pd.datetime.now() + pd.DateOffset(seconds = 10)#10초 뒤
pd.datetime.now() + pd.DateOffset(microseconds = 10)
pd.datetime.now() - pd.Timestamp('2019-6-18')
(pd.datetime.now() - pd.Timestamp('2019-6-18')).days

pd.datetime.now() + pd.Timedelta('8:00:00')
pd.datetime.now() + pd.Timedelta('1days 8:00:00')
pd.datetime.now() + pd.Timedelta('1days 8hours')
pd.datetime.now() - pd.Timedelta('1days')
pd.datetime.now() + pd.Timedelta('- 1days')
pd.datetime.now() + pd.Timedelta('- 1days 2 min')

pd.Timestamp('2019-9-23') + pd.DateOffset(year = 2018)#년도가 2018년으로 수정됨
pd.Timestamp('2019-9-23') + pd.DateOffset(
        month = 6, day = 18, hour = 9, minute = 30)

#Series에서 시간함수를 사용해보자 dt를 이용한다. 
pd.to_datetime('2019-9-23').year
pd.to_datetime('2019-9-23').dt.year#error
#dt.year는 Series에서만 사용된다.
Series(pd.to_datetime('2019-9-23')).dt.year

pd.to_datetime('2019-9-23').month
pd.to_datetime('2019-9-23').day
pd.to_datetime('2019-9-23').day_name()
pd.to_datetime('2019-9-23').weekday() #정수형 요일 월:0 ~ 일:6
pd.to_datetime('2019-9-23').quarter#()하면 error


pd.to_datetime('2019-9-23').strftime('%Y')

Series(pd.to_datetime('2019-9-23')).dt.year
Series(pd.to_datetime('2019-9-23')).dt.month
Series(pd.to_datetime('2019-9-23')).dt.day
Series(pd.to_datetime('2019-9-23')).dt.hour
Series(pd.to_datetime('2019-9-23')).dt.minute
Series(pd.to_datetime('2019-9-23')).dt.second
Series(pd.to_datetime('2019-9-23')).dt.microsecond
Series(pd.to_datetime('2019-9-23')).dt.weekday_name#요일 이름 출력
Series(pd.to_datetime('2019-9-23')).dt.weekday#요일 숫자 출력 
Series(pd.to_datetime('2019-9-23')).dt.day_name()
Series(pd.to_datetime('2019-9-23')).dt.year
Series(pd.to_datetime('2019-9-23')).dt.is_month_end
Series(pd.to_datetime('2019-9-30')).dt.is_month_end
Series(pd.to_datetime('2019-9-23')).dt.is_month_start
Series(pd.to_datetime('2019-9-23')).dt.daysinmonth#이달의 날짜 수 

#[문제 138] 2003년도 이전에 입사한 사원들의 정보를 출력하시오 
emp = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\emp.csv')
emp
emp.columns
emp.info()
emp['HIRE_DATE']
pd.to_datetime(emp['HIRE_DATE']) < pd.to_datetime('2003')
emp[pd.to_datetime(emp['HIRE_DATE']) < pd.to_datetime('2003')]


#[문제 139] 2005년도에 입사한 사원들의 정보를 출력하시오
emp[pd.to_datetime(emp['HIRE_DATE']).dt.year == 2005]


#[문제 140] 년도별 총액 급여를 출력하세요
emp.groupby(pd.to_datetime(emp['HIRE_DATE']).dt.year)['SALARY'].sum()

#[문제 141] 2005년도에 입사한 사원들의 분기별 급여 총액을 구하시오
emp_2005 = emp[pd.to_datetime(emp['HIRE_DATE']).dt.year == 2005]

emp_2005.groupby(pd.to_datetime(emp_2005['HIRE_DATE']).dt.quarter)['SALARY'].sum()

#[문제 142] 요일별 입사인원수를 출력해주세요
pd.to_datetime(emp['HIRE_DATE']).dt.weekday_name
emp['EMPLOYEE_ID'].groupby(
        pd.to_datetime(emp['HIRE_DATE']).dt.weekday_name).count()
#HIRE_DATE
#Friday       19
#Monday       10
#Saturday     19
#Sunday       15
#Thursday     16
#Tuesday      13
#Wednesday    15
#Name: EMPLOYEE_ID, dtype: int64
emp['EMPLOYEE_ID'].groupby(
        pd.to_datetime(emp['HIRE_DATE']).dt.weekday).count()
week = emp['EMPLOYEE_ID'].groupby(
        pd.to_datetime(emp['HIRE_DATE']).dt.weekday).count()
week.index = [['월','화','수','목','금','토','일']]
week
#월    10
#화    13
#수    15
#목    16
#금    19
#토    19
#일    15
#Name: EMPLOYEE_ID, dtype: int64

#[문제 143] 년도, 분기별 급여의 총액을 구하세요

pd.to_datetime(emp['HIRE_DATE']).dt.year
pd.to_datetime(emp['HIRE_DATE']).dt.quarter

emp['SALARY'].groupby([pd.to_datetime(emp['HIRE_DATE']).dt.year,
   pd.to_datetime(emp['HIRE_DATE']).dt.quarter]).sum()

emp['SALARY'].groupby([pd.to_datetime(emp['HIRE_DATE']).dt.year,
   pd.to_datetime(emp['HIRE_DATE']).dt.quarter]).sum().unstack()


hire_year = pd.to_datetime(emp['HIRE_DATE']).dt.year
hire_quar = pd.to_datetime(emp['HIRE_DATE']).dt.quarter
result = emp.groupby([hire_year, hire_quar])['SALARY'].sum().unstack()
result.columns = ['1분기','2분기','3분기','4분기']
result.index.name = '년도'
result
#          1분기      2분기      3분기      4분기
#년도                                      
#2001  17000.0      NaN      NaN      NaN
#2002      NaN  36808.0  21008.0  11000.0
#2003      NaN  35000.0   8000.0   3500.0
#2004  40700.0  14300.0  17000.0  14000.0
#2005  86900.0  16800.0  60800.0  33400.0
#2006  69400.0  20400.0  14200.0  17100.0
#2007  36600.0  20200.0   2500.0  35600.0
#2008  46900.0  12300.0      NaN      NaN


#
#sqlite library
#별도의 DB서버가 필요 없이 DB파일을 기초하여 데이터베이스 처리하는 엔진
import sqlite3
dir(sqlite3)
#메모리상에 sqlite 구성하기 
conn = sqlite3.connect(":memory:")
#cursor 설정, 객체 지정 
c = conn.cursor()

c.execute('create table emp(id integer, name char, sal integer)')

c.execute("insert into emp(id,name,sal) values (1,'한조',1000)")

c.execute("select * from emp")#결과는 메모리상에만 있다. 불러와야 한다.
#(pl/sql 할때 명시적 커서는 fetch 단계를 거처야 한다.) 
c.fetchone()
#(1, '한조', 1000)

c.execute("insert into emp(id,name,sal) values (2,'겐지',1100)")
c.execute("select * from emp")
#한 줄씩 출력된다. 
c.fetchone()#(1, '한조', 1000)
c.fetchone()#(2, '겐지', 1100)
c.fetchone()#

c.execute("select * from emp")
#전체가 출력된다. list안에 tuple로 들어가 있다. 
c.fetchall()#[(1, '한조', 1000), (2, '겐지', 1100)]

c.execute("select * from emp")
lst = c.fetchall()
type(lst)
for i in lst:
    print(i)
    
#rollback과 commit들은 cursor를 기반으로 하면 안 된다. connect 기반이다. 
conn.rollback() #c.rollback()이 아님에 주의하자 

#다시 insert를 한 후에 
c.execute("insert into emp(id,name,sal) values (1,'한조',1000)")
c.execute("insert into emp(id,name,sal) values (2,'겐지',1100)")
c.execute("select * from emp")
c.fetchall()

#cursor를 닫고, 연결도 닫으면 메모리에 있던 table 정보가 사라진다. 
c.close()
conn.close()
#
conn = sqlite3.connect(":memory:")
c = conn.cursor()
c.execute("select * from emp")#OperationalError: no such table: emp
#sqlite안에 table들을 조회해 보자 
c.execute("select name from sqlite_master where type = 'table'")
c.fetchall()#[]

c.close()
conn.close()

#데이터를 영구저장을 하고 싶을때 하드디스크에 DB파일을 저장한다. 
conn = sqlite3.connect("C:\\WorkSpace\\Python_Space\\data\\insa.db")
c = conn.cursor()
c.execute("select name from sqlite_master where type = 'table'")
c.fetchall()

c.execute('create table emp(id integer, name char, sal integer)')
c.execute("insert into emp(id,name,sal) values (1,'한조',1000)")
c.execute("insert into emp(id,name,sal) values (2,'겐지',1100)")
c.execute("select * from emp")
c.fetchall()
conn.commit()
c.close()
conn.close()

conn = sqlite3.connect("C:\\WorkSpace\\Python_Space\\data\\insa.db")
c = conn.cursor()
c.execute("select name from sqlite_master where type = 'table'")
c.fetchall() #[('emp',)]
c.execute("drop table emp")
c.execute("select name from sqlite_master where type = 'table'")
c.fetchall() #[]

#csv file을 불러와서 sqlite에 넣어보자 
data = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\emp.csv")
#to_sql(pandas 의 명령어) : emp라는 table로 저장, index는 제외
data.to_sql('emp',conn, index = False)

c.execute("select * from emp")
c.fetchall()
#column의 정보를 확인하자 
c.execute("PRAGMA table_info(emp)")
c.fetchall()#column의 이름과 데이터 타입을 확인할 수 있다. 

c.execute('create table emp2(id integer, name char, sal integer)')
c.execute("insert into emp2(id,name,sal) values (1,'한조',1000)")
c.execute("insert into emp2(id,name,sal) values (2,'겐지',1100)")
c.execute("select * from emp2")
c.fetchall()
conn.commit()

#sqlite에서 pandas로 옮겨보자 
emp2 = pd.read_sql_query("select * from emp2", conn)
emp2
c.execute("select * from emp where upper(last_name) = 'KING'")
c.fetchall()
c.execute("select * from emp where upper(last_name) = 'king'")
c.fetchall()
c.execute("select last_name from emp where upper(last_name) = 'KING'")
c.fetchall()
#substr() : 문자를 추출하는 함수, substr(문자열,처음위치,추출갯수)
c.execute("select last_name, substr(last_name,1,2), substr(last_name,-2,2) from emp")
c.fetchone()
#replace(문자열,바꿀문자,바뀐문자) : 문자를 치환하는 함수
c.execute("select replace('데이터 분석가','분석가','과학자')")
c.fetchone()
#trim(문자 from 문자) : 접두, 접미 문자를 자르는 함수
c.execute("select trim('   데이터 분석가    ')")
c.fetchone()

c.execute("select ltrim('     데이터 분석가    ')")
c.fetchone()

c.execute("select rtrim('    데이터 분석가     ')")
c.fetchone()
#instr() : 문자의 (처음 나오는)위치를 반환하는 함수(대소문자 구분 주의)
#instr(문자열, 찾는 문자, 시작위치, 나온 순서)
c.execute("select instr('aaa*@&#*@&@$$dfoejf','@')")
c.fetchone()

c.execute("select upper(substr(last_name,1,1)) || lower(substr(last_name,2)) from emp")
c.fetchall()    

c.execute("select 1+1, 1-2, 4*3, 11/2, 32%5")
c.fetchone()
#
c.execute("select last_name, salary*12, + ifnull(commission_pct, 0) from emp")
c.fetchall()
#nvl은 여기서는 안 된다.
c.execute("select last_name, nvl(commission_pct, 0) from emp")#error 
c.fetchall()
#coalesce(A,B) 널이 아닌거 출력, 다 널이면 널 출력, 앞에 입력된거부터 null인지 봄
c.execute("select last_name, coalesce(commission_pct, 0) from emp")
c.fetchall()
#nullif() : null 만듬, 두 값이 같으면 null 출력
c.execute("select last_name, nullif(length(last_name), 5) from emp")
c.fetchall()
#date('now') 는 sql에서 sysdate임 
c.execute("select date('now'), datetime('now','localtime')")
c.fetchall()

c.execute("select date('now','91 day'), date('now','-97 day')")
c.fetchall()

c.execute("select date('now','2 month')")
c.fetchone()

c.execute("select date('now','1 year','2 month','10 day')")
c.fetchall()

c.execute("select datetime('now','localtime','2 hours','60 minute')")
c.fetchone()

c.execute("select date('2019-09-23', 'weekday 5')")
c.fetchone()

c.execute("select date('now') - date(hire_date) from emp")
c.fetchall()

c.execute("select distinct department_id from emp")
c.fetchall()

c.execute("select * from emp where employee_id in (100,200)")
c.fetchall()

c.execute("select * from emp where last_name like '%d'")
c.fetchall()

c.execute("select * from emp where salary >= 15000 and salary <= 20000")
c.fetchall()

c.execute("select * from emp where salary between 15000 and 20000")
c.fetchall()

c.execute("select * from emp where salary >= 15000 or job_id = 'SA_REP'")
c.fetchall()

c.execute("select * from emp where employee_id in \
          (select manager_id from emp)")
c.fetchall()

c.execute("select * from emp e where exists \
          (select 'x' from emp where manager_id = e.employee_id)")
c.fetchall()

#not in 은 all 이라서 null값 있으면 결과 없음 
c.execute("select * from emp where employee_id not in \
          (select manager_id from emp where manager_id is not null)")
c.fetchall()

c.execute("select * from emp e where not exists \
          (select 'x' from emp where manager_id = e.employee_id)")
c.fetchall()

c.execute("select * from emp where commission_pct is null")
c.fetchall()

c.execute("select * from emp where commission_pct is not null")
c.fetchall()
#sql할때는 to_char(), to_date() 등을 사용했다. 여기서는 되지 않는다.
c.execute("select strftime('%Y',date(hire_date)) from emp")
c.fetchall()

c.execute("select strftime('%Y %m %d %H %M %S','now')")
c.fetchall()

#cast() : 형변환함수 
c.execute("select salary, cast(salary as int), cast(salary as text) from emp")
c.fetchall()

c.execute("select cast('1.24' as real) + 1, cast('1.24' as decimal) + 1")
c.fetchone()

c.execute("""select last_name, salary, commission_pct, 
          case
              when commission_pct is null then salary * 12
              else (salary * 12 + (salary * 12 * commission_pct))
          end
          from emp""")
c.fetchall()

#[문제 144] 아래와 같이 출력해주세요
c.execute("""select count(*) ,
          count(case substr(hire_date,1,4) when '2001' then 1 end),
          count(case substr(hire_date,1,4) when '2002' then 1 end),
          count(case substr(hire_date,1,4) when '2003' then 1 end),
          count(case substr(hire_date,1,4) when '2004' then 1 end),
          count(case substr(hire_date,1,4) when '2005' then 1 end),
          count(case substr(hire_date,1,4) when '2006' then 1 end),
          count(case substr(hire_date,1,4) when '2007' then 1 end),
          count(case substr(hire_date,1,4) when '2008' then 1 end)
          from emp""")
x = c.fetchall()
print('전체인원수',x[0][0])
print('2001 인원수',x[0][1])

c.execute("""
          select 
          sum(case substr(hire_date,1,4) when '2001' then 1 else 0 end) "2001",
          sum(case substr(hire_date,1,4) when '2002' then 1 else 0 end) "2002",
          sum(case substr(hire_date,1,4) when '2003' then 1 else 0 end) "2003",
          sum(case substr(hire_date,1,4) when '2004' then 1 else 0 end) "2004",
          sum(case substr(hire_date,1,4) when '2005' then 1 else 0 end) "2005",
          sum(case substr(hire_date,1,4) when '2006' then 1 else 0 end) "2006",
          sum(case substr(hire_date,1,4) when '2007' then 1 else 0 end) "2007",
          sum(case substr(hire_date,1,4) when '2008' then 1 else 0 end) "2008"
          from emp
          """)
x = c.fetchall()
x

#decode는 안 된다.
c.execute("""
          select 
          max(decode(yy,'2001',co)) "2001",
          max(decode(yy,'2002',co)) "2002",
          max(decode(yy,'2003',co)) "2003",
          max(decode(yy,'2004',co)) "2004",
          max(decode(yy,'2005',co)) "2005",
          max(decode(yy,'2006',co)) "2006",
          max(decode(yy,'2007',co)) "2007",
          max(decode(yy,'2008',co)) "2008"
          from
          (select substr(hire_date,1,4) yy, count(*) co 
           from emp
           group by substr(hire_date,1,4));
          """)
c.fetchall()

#pivot는 안 되나 ???? 
c.execute("""
          select *
          from (select substr(hire_date,1,4) yyyy 
                from emp)
          pivot(count(*) for yyyy in ('2001','2002','2003','2004',
          '2005','2006','2007','2008'))
          """)
c.fetchall()

#부서별 salary 총 합
c.execute("""
          select department_id, sum(salary)
          from emp
          group by department_id
          having sum(salary) > 100000""")
c.fetchall()
