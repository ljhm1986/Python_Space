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

########################################################################
#9/24#
from pandas import Series, DataFrame
import pandas as pd
import sqlite3
conn = sqlite3.connect("C:\\WorkSpace\\Python_Space\\data\\insa.db")
c = conn.cursor()
#
c.execute("select name from sqlite_master where type = 'table'")
c.fetchall()#[] 있다면 지워주고 시작 

c.execute("create table emp_1(id integer, name text, sal integer)")
c.execute("select name from sqlite_master where type = 'table'")
c.fetchall()
c.execute("insert into emp_1(id, name, sal) values (1,'로드호그',1000)")
c.fetchone()
conn.commit()
c.close()
conn.close()

#
conn = sqlite3.connect("C:\\WorkSpace\\Python_Space\\data\\insa.db")
c = conn.cursor()
c.execute("select name from sqlite_master where type = 'table'")
c.fetchall()
c.execute("select * from emp_1")
c.fetchall()
#[(1, '로드호그', 1000)]
c.execute("insert into emp_1(id,name,sal) values(?,?,?)",\
          (2,'라인하르트',2000))
c.execute("select * from emp_1")
c.fetchall()
#[(1, '로드호그', 1000), (2, '라인하르트', 2000)]

#?: 값을 넣으려는곳의 자리표시자, 넣으려는 값은 다음 tuple에서 나타난다.
insert_sql = "insert into emp_1(id, name, sal) values (?,?,?)"
c.execute(insert_sql,(3,'겐지',3000))
c.execute("select * from emp_1")
c.fetchall()#[(1, '로드호그', 1000), (2, '라인하르트', 2000), (3, '겐지', 3000)]
conn.commit()

c.execute("select * from emp_1")
c.fetchmany(2)#2개만 출력하자
#[(1, '로드호그', 1000), (2, '라인하르트', 2000)]

c.execute("update emp_1 set sal = 5000 where id = 1")
c.execute("select * from emp_1 where id = 1")
c.fetchone()
conn.rollback()
c.execute("select * from emp_1 where id = 1")
c.fetchone()
c.execute("delete from emp_1 where id = 1")
c.execute("select * from emp_1")
c.fetchall()
conn.rollback()
c.execute("select * from emp_1")
c.fetchall()
c.execute("alter table emp_1 add column deptno integer")
#table의 정보를 보자 
c.execute("PRAGMA table_info(emp_1)")
c.fetchall()
#[(0, 'id', 'integer', 0, None, 0),
# (1, 'name', 'text', 0, None, 0),
# (2, 'sal', 'integer', 0, None, 0),
# (3, 'deptno', 'integer', 0, None, 0)]
c.execute("select name from sqlite_master where type = 'table'")
c.fetchall()
c.execute("drop table emp_1")
c.execute("select name from sqlite_master where type = 'table'")
c.fetchall()#[]

#table 2개를 만들자 
c.execute("create table emp(id integer, name text, sal integer, deptno integer)")
c.execute("insert into emp(id, name, sal, deptno) values(1,'루시우',1000,10)")
c.execute("insert into emp(id, name, sal, deptno) values(2,'솔져',1500,20)")
c.execute("insert into emp(id, name, sal, deptno) values(3,'리퍼',2000,30)")
c.execute("insert into emp(id, name, sal, deptno) values(4,'디바',2500,40)")
c.execute("insert into emp(id, name, sal, deptno) values(5,'겐지',3000,50)")
c.execute("select * from emp")
c.fetchall()
conn.commit()

c.execute("create table dept(deptno integer, dname text)")
c.execute("insert into dept(deptno,dname) values(10,'힐러부')")
c.execute("insert into dept(deptno,dname) values(20,'공격부')")
c.execute("insert into dept(deptno,dname) values(30,'선발부')")
c.execute("insert into dept(deptno,dname) values(40,'탱커부')")
c.execute("insert into dept(deptno,dname) values(50,'딜러충부')")
c.execute("select * from dept")
c.fetchall()
conn.commit()

#join : 두 개 이상의 table에서 데이터를 가져오는 방법
#1. cartesian product : join조건이 없을때, 조건이 잘못되었을떄 
#각 테이블 행을 모두 곱함 
c.execute("select id,name,dname from emp,dept")
c.fetchall()
c.execute("select id,name,dname from emp cross join dept")
c.fetchall()

#2.equi join, inner join, simple join, 등가조인
c.execute("""
          select id, name, e.deptno, dname
          from emp e, dept d
          where e.deptno = d.deptno""")
c.fetchall()
#ANSI 표준으로는 join on, inner join, join using 
c.execute("""
          select id, name, e.deptno, dname
          from emp e join dept d on e.deptno = d.deptno""")
c.fetchall()
c.execute("""
          select id, name, e.deptno, dname
          from emp e inner join dept d on e.deptno = d.deptno""")
c.fetchall()
c.execute("""
          select id, name, deptno, dname
          from emp e join dept d using(deptno)
          """)
c.fetchall()
#natural join 조인조건을 자동으로 만듬
c.execute("""
          select id, name, deptno, dname
          from emp e natural join dept d
          """)
c.fetchall()
#3. outer join
c.execute("""
          select id, name, e.deptno, dname
          from emp e left outer join dept d 
          on e.deptno = d.deptno
          """)
c.fetchall()
#sqlite에서는 right outer join은 지원하지 않는다.
c.execute("""
          select id, name, e.deptno, dname
          from emp e right outer join dept d 
          on e.deptno = d.deptno
          """)
#OperationalError: RIGHT and FULL OUTER JOINs are not currently supported
c.fetchall()

#full outer join 하려면 두개의 left outer join을 union 한다.
c.execute("""
          select id, name, e.deptno, dname
          from emp e left outer join dept d 
          on e.deptno = d.deptno
          union
          select id, name, e.deptno, dname
          from dept d left outer join emp e
          on e.deptno = d.deptno
          """)
c.fetchall()
#중복을 체크하지 않고 합치려면 unoin all 
c.execute("""
          select id, name, e.deptno, dname
          from emp e left outer join dept d 
          on e.deptno = d.deptno
          union all
          select id, name, e.deptno, dname
          from dept d left outer join emp e
          on e.deptno = d.deptno
          """)
c.fetchall()
#교집합을 구하려면 insersect
c.execute("""
          select id, name, e.deptno, dname
          from emp e left outer join dept d 
          on e.deptno = d.deptno
          intersect
          select id, name, e.deptno, dname
          from dept d left outer join emp e
          on e.deptno = d.deptno
          """)
c.fetchall()
#차집합을 구하려면 except
c.execute("""
          select id, name, e.deptno, dname
          from emp e left outer join dept d 
          on e.deptno = d.deptno
          except
          select id, name, e.deptno, dname
          from dept d left outer join emp e
          on e.deptno = d.deptno
          """)
c.fetchall()

#4.non equi join
#새로운 table(job_grades) 을 만들자  
c.execute('drop table job_grades')
c.execute("""
          create table job_grades
          (grade_level, text, lowest_sal integer, highest_sal integer)
          """)
c.execute("""
          insert into job_grades
          (grade_level, lowest_sal, highest_sal) values('A', 1000, 2999)
          """)
c.execute("""
          insert into job_grades
          (grade_level, lowest_sal, highest_sal) values('B', 3000, 6999)
          """)
c.execute("""
          insert into job_grades
          (grade_level, lowest_sal, highest_sal) values('C', 6000, 9999)
          """)
c.execute("""
          insert into job_grades
          (grade_level, lowest_sal, highest_sal) values('D', 10000, 14999)
          """)
c.execute("""
          insert into job_grades
          (grade_level, lowest_sal, highest_sal) values('F', 15000, 29990)
          """)
c.execute("select * from job_grades")
c.fetchall()

#csv file을 불러와서 db파일에 저장하자 
data = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\emp.csv")
data.to_sql('employees',conn,index = False)
c.execute('PRAGMA table_info(employees)')
c.fetchall()

#직원들의 급여에 등급을 매기자 
c.execute("""
          select e.employee_id, e.salary, j.grade_level
          from employees e join job_grades j
          on e.salary between j.lowest_sal and j.highest_sal
          """)
c.fetchall()

#5.self join
#직원과 해당 상급자 
c.execute("""
          select e1.employee_id, e1.last_name, e2.employee_id, e2.last_name
          from employees e1 join employees e2
          on e1.manager_id = e2.employee_id
          """)
c.fetchall()
c.execute("""
          select e1.employee_id, e1.last_name, e2.employee_id, e2.last_name
          from employees e1 left outer join employees e2
          on e1.manager_id = e2.employee_id
          """)
c.fetchall()

#sqlite 에서 group 함수
c.execute("""
          select sum(salary), avg(salary), max(salary),
          min(salary), count(*), count(department_id)
          from employees
          """)
c.fetchall()

#부서번호와 직업번호와 급여총합이 10000 넘는 것만 출력 
c.execute("""
          select department_id, job_id, sum(salary)
          from employees
          group by department_id
          having sum(salary) > 10000
          order by department_id asc, job_id
          """)
c.fetchall()
#group_concat : 포함된것들을 모두 나열하여 보여줌 
c.execute("""
          select department_id, group_concat(last_name)
          from employees
          group by department_id
          """)
c.fetchall()
#tuple 2번째 칸에 last_name들이 한 문자열로 나열되어 있다. 
c.execute("""
          select department_id, group_concat(distinct job_id)
          from employees
          group by department_id
          """)
c.fetchall()
#[(None, 'SA_REP'),
# (10.0, 'AD_ASST'),
# (20.0, 'MK_MAN,MK_REP'),
# (30.0, 'PU_MAN,PU_CLERK'),
# (40.0, 'HR_REP'),
# (50.0, 'ST_MAN,ST_CLERK,SH_CLERK'),
# (60.0, 'IT_PROG'),
# (70.0, 'PR_REP'),
# (80.0, 'SA_MAN,SA_REP'),
# (90.0, 'AD_PRES,AD_VP'),
# (100.0, 'FI_MGR,FI_ACCOUNT'),
# (110.0, 'AC_MGR,AC_ACCOUNT')]

#자신이 속한 부서의 평균임금보다 임금이 높은 직원은?
#inline view 
c.execute("""
          select e1.employee_id, e1.last_name, e1.salary
          from employees e1, 
          (select department_id, avg(salary) avg_sal
          from employees
          group by department_id) e2
          where e1.salary >= e2.avg_sal and
          e1.department_id = e2.department_id
          order by 1
          """)       
c.fetchall()
#상호관련서브쿼리 
c.execute("""
          select e1.employee_id, e1.last_name, e1.salary
          from employees e1
          where salary >=
          (select avg(salary)
          from employees
          where department_id = e1.department_id)
          order by 1
          """)
c.fetchall()       

#[문제 145] 2006년도에 입사한 사원들의 부서이름별로 급여의 총액, 평균을 구하세요.
data2 = pd.read_csv("C:\\WorkSpace\\Python_Space\\data\\dept.csv")
data2
data2.to_sql('departments',conn,index = False)
conn = sqlite3.connect("C:\\WorkSpace\\Python_Space\\data\\insa.db")
c = conn.cursor()
c.execute("select name from sqlite_master where type = 'table'")
c.fetchall()
c.execute("select * from emp")
c.fetchall()
c.execute("select * from employees")
c.fetchall()
c.execute("select * from departments")
c.fetchall()
c.execute("PRAGMA table_info(employees)")
c.fetchall()
c.execute("""
          select e.sum_sal, e.avg_sal, d.department_name
          from departments d,
          (
          select department_id, round(sum(salary)) sum_sal,
          round(avg(salary)) avg_sal
          from employees
          where substr(hire_date,1,4) = '2006'
          group by department_id) e
          where d.department_id = e.department_id
          """)
c.fetchall()

#[문제 146] 사원들의 employee_id, salary, department_name, grade_level
#을 출력해주세요
c.execute("select * from job_grades")
c.fetchall()

c.execute("""
          select 
          e.employee_id, e.salary, j.grade_level, d.department_name
          from employees e left outer join departments d
          on e.department_id = d.department_id
          left outer join job_grades j        
          on (e.salary between j.lowest_sal and j.highest_sal)
          """)
c.fetchall()

#[문제 147] last_name에 a가 2개 이상 포함되어 있는 사원들의 
#employee_id, last_name, salary, department_name, grade_level을 
#출력하세요
c.execute("""
          select employee_id, last_name,salary
          from employees
          where last_name like '%a%a%'
          """)
c.fetchall()

c.execute("""
          select 
          e.employee_id, e.salary, j.grade_level, d.department_name
          from employees e, job_grades j, departments d
          where (e.salary between j.lowest_sal and j.highest_sal)
          and (e.department_id = d.department_id)
          and last_name like '%a%a%'
          """)
c.fetchall()

#[문제 148] 회사에서 최고 급여를 받는 사원들의 employee_id, salary,
#department_name을 출력하세요
c.execute("""
          select e1.employee_id, e1.salary,d.department_name
          from employees e1,
          (select max(salary) max_sal
          from employees) e2, departments d
          where e1.salary == e2.max_sal
          and e1.department_id = d.department_id
          """)
c.fetchall()

#[문제 149] King에게 보고하는 사원들의 last_name, salary를 출력하세요
#1.subquery
c.execute("""
          select e.last_name, e.salary
          from employees e
          where e.manager_id in
          (select employee_id
          from employees
          where upper(last_name) == 'KING')
          order by 1
          """)
c.fetchall()          
#2.join
c.execute("""
          select e1.last_name, e1.salary
          from employees e1, employees e2
          where e1.manager_id = e2.employee_id
          and upper(e2.last_name) = 'KING'
          """)
c.fetchall() 
##이제 sql은 여기까지 하고 ~


#numpy 부분은 numpy.py로 이동 

##############################################################################
#9/25#
#시각화를 해 보자
#matplotlib - 시각화 패키지
import matplotlib.pylab as plt
labels = ['삼성전자','SK하이닉스','LG전자','네이버','카카오']
ratio = [50,20,10,10,10]

plt.pie(ratio, labels = labels)
plt.show()
#한글이 깨져서 나온다.
#폰트에 한글폰트를 설정하자 
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)
plt.pie(ratio, labels = labels)
plt.show()

#pie(x(array): data 조각 크기, labels(list,) : 각 x의 label,
# explode(array): 각 x가 중심에서 떨어질 거리, 
# colors(array) : 색상, autopct(string, function) : 각 x의 안에 넣는 값
# textprops(dict) : text 설정)

colors = ['gold','yellowgreen','lightcoral','lightskyblue','red']
explode = [0.0,0.1,0.0,0.0,0.0]
plt.pie(ratio,
        labels = labels,
        explode = explode,
        colors = colors)
plt.show()

plt.pie(ratio, labels = labels, explode = explode,
        colors = colors, 
        autopct = '%1.1f%%')#전체대비 비율값이 들어간다. 
plt.show()


plt.figure(figsize = (10,10))
plt.pie(ratio, labels = labels, explode = explode,
        colors = colors, autopct = '%1.1f%%',
        textprops = {'fontsize': 20})#글자 크기가 바뀐다.

#[문제 151] blood.csv file을 읽어들여서 도수분포표 작성 
blood = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\blood.csv')
blood
blood1 = DataFrame(blood)
blood1
blood1['GENDER'].value_counts()
x = blood1['BLOODTYPE'].value_counts()
x

plt.pie(x, labels = x.index, colors = colors,
        autopct = '%1.1f%%',
        textprops = {'fontsize':20})
plt.title("혈액형 현황",fontsize = 20)
plt.show()

#emp 를 불러와서 그래프를 그려보자 
emp = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\emp.csv')
emp.info()
emp['DEPARTMENT_ID'].value_counts()
emp['DEPARTMENT_ID'].value_counts(dropna = False)
#부서별 인원 그래프
y = emp['DEPARTMENT_ID'].value_counts()

plt.figure(figsize = (10,10))
plt.pie(y, labels = y.index, colors = colors, autopct = '%1.1f%%',
        textprops = {'fontsize':20})
plt.title("부서별 인원 현황",fontsize = 20)
plt.show()

#직업별 인원 그래프
y = emp['JOB_ID'].value_counts()

plt.figure(figsize = (10,10))
plt.pie(y, labels = y.index, colors = colors, autopct = '%1.1f%%',
        textprops = {'fontsize':20})
plt.title("직업별 인원 현황",fontsize = 20)
plt.show()

#부서별 인원, 부서가 없는 사원도 고려해보자 
z = ['부서x' if pd.isnull(i) else str(int(i))+'부서' 
     for i in emp['DEPARTMENT_ID']]
z = Series(z).value_counts()
z

plt.figure(figsize = (10,10))
plt.pie(z, labels = z.index, colors = colors, autopct = '%1.1f%%',
        textprops = {'fontsize':20})
plt.title("부서별 인원 현황",fontsize = 20)
plt.show()

##Counter()를 이용해보자 
import collections
z = ['부서x' if pd.isnull(i) else str(int(i))+'부서' 
     for i in emp['DEPARTMENT_ID']]
cnt = collections.Counter(z)
cnt.values()
cnt.keys()
#
plt.figure(figsize = (10,10))
plt.pie(cnt.values(), labels = cnt.keys(),
        colors = colors, autopct = '%1.1f%%',
        textprops = {'fontsize':20})
plt.title("부서별 인원 현황",fontsize = 20)
plt.show()

##막대그래프를 만들어 보자 ##
x = ['SQL','R','PLSQL']
y = [90,78,65]
plt.bar(x,y, color=['orange','green','blue'])
plt.title('과목별 점수 현황',
          fontsize = 20)#font로도 가능 
plt.xlabel('과목', size = 15)
plt.ylabel('점수', size = 15)
plt.grid(True)#격자선 
plt.show()

#
xlabel = ['SQL','R','PLSQL']
x = [1,2,3]
y = [90,78,65]
plt.bar(x,y, color=['orange','green','blue'])
plt.title('과목별 점수 현황', fontsize = 20)
#xticks(ticks(array 같은거) : 위치의 list들,
#labels(array 같은거) : 주여진 위치에 나타내고 싶은 이름들)
plt.xticks(x,xlabel)
#이렇게 하면 x축에 1,2,3이 나오는게 아니라 SQL, R, PLSQL이 나온다.
plt.xlabel('과목', size = 15)
plt.ylabel('점수', size = 15)
plt.grid(True)
plt.show()

#barh() : 막대가 누워서 만들어진다. 
plt.barh(x,y,color=['orange','green','blue'], alpha = 0.8)
plt.title('과목별 점수 현황', fontsize = 20)
plt.yticks(x,xlabel)
plt.ylabel('과목', size = 15)
plt.xlabel('점수', size = 15)
plt.grid(True)
plt.show()

#부서별 인원을 구해서 막대 그래프를 그려보자 
x = emp['DEPARTMENT_ID'].index
y = emp['DEPARTMENT_ID'].value_counts(dropna = False)
type(y)#Series
y.index
y.values
y1 = y.index
y1
#Float64Index([50.0, 80.0, 30.0, 100.0, 60.0, 90.0, 110.0, 20.0, 70.0, 40.0,
#              10.0, nan],dtype='float64')
y2 = y.values
y2
#array([45, 34,  6,  6,  5,  3,  2,  2,  1,  1,  1,  1], dtype=int64)
y1 = y1.fillna(-1)#'부서x'로 하니 object 형이라 안된다고 error
plt.figure(figsize = (6,6))
plt.bar(y1, y2, width =7)#width=0.8이 기본값, 너무 가늘었음 
plt.title('부서별 인원 현황', fontsize = 20)#font로도 가능 
plt.xlabel('부서번호', size = 15)
plt.ylabel('인원', size = 15)
plt.grid(True)
plt.show()
#
plt.figure(figsize = (6,6))
plt.barh(y1,y2,alpha = 0.8,#alpha 는 투명도 
         height = 7)
##list 내장 객체 사용
y1 = y.index
y2 = y.values
y1 = ['부서없음' if pd.isnull(i) else str(int(i))+'부서' 
      for i in y.index]
plt.figure(figsize = (8,8))
plt.bar(y1, y2)
plt.title('부서별 인원 현황', fontsize = 20)#font로도 가능 
plt.xlabel('부서번호', size = 15)
plt.ylabel('인원', size = 15)
plt.grid(True)
plt.show()
##Counter 사용 
z = ['부서없음' if pd.isnull(i) else str(int(i))+'부서' 
     for i in emp['DEPARTMENT_ID']]
z = collections.Counter(z)
z.keys()
z.values()

plt.figure(figsize = (8,8))
plt.bar(z.keys(), z.values())
plt.title('부서별 인원 현황', fontsize = 20)#font로도 가능 
plt.xlabel('부서번호', size = 15)
plt.ylabel('인원', size = 15)
plt.grid(True)
plt.show()

##다른 방법으로는 
z = ['부서없음' if pd.isnull(i) else str(int(i))+'부서' 
     for i in emp['DEPARTMENT_ID']]
z = Series(z).value_counts()
#다음과 같이 종류만 입력해서 그릴 수 있다. 
z.plot(kind = 'bar')
z.plot(kind = 'barh')

#선 그래프를 그려보자 
#plot(**args, scalex=True, scaley=True, data=None, **kwargs)
plt.plot([0,5,10,15,20,25,30,20,5])#y축 값, x축 값은 자동으로 할당된다. 
plt.plot([100,120,150,500,600,800],#x축 값
         [1,5,7,9,15,33],#y축 값
         color = 'y') #r,g,b,c,m,y,k,w 등의 약자들이 있다. 
#r = red, g = green, b = blue, c = cyan, m = magenta, y = yellow,
#k = black, w = white
plt.plot([100,120,150,500,600,800],[1,5,7,9,15,33],
         color = 'k')
plt.plot([100,120,150,500,600,900],[1,5,7,9,15,33],
         color = '0.75')
plt.plot([10,12,15,50,60,80],[1,5,7,9,15,33],
         color = 'b', linestyle = 'dotted')
#linestyle = 'dotted','solid','dashed','dashot'
plt.plot([100,200,300,400,500],[1,5,10,15,20],'-g')#직선이 된다
plt.plot([100,200,300,400,500],[1,5,10,15,20],'--c')#점선이 된다.
plt.plot([100,200,300,400,500],[1,5,10,15,20],'-.k')#반점선이 된다.
plt.plot([100,200,300,400,500],[1,5,10,15,20],':r')#점선이 된다
plt.plot([100,200,300,400,500],[1,5,10,15,20],'r1--')#

data = {"도바킨":[15,13,11],"게롤드":[13,14,15],"프랭클린":[10,9,12]} 
data
df = DataFrame(data, index = [2015,2016,2017])
df
#      도바킨  게롤드  프랭클린
#2015   15   13    10
#2016   13   14     9
#2017   11   15    12
df.rank()
df.rank(axis = 0)#열별 순위, 기본값
#      도바킨  게롤드  프랭클린
#2015  3.0  1.0   2.0
#2016  2.0  2.0   1.0
#2017  1.0  3.0   3.0
df.rank(axis = 1)#행별 순위 
#      도바킨  게롤드  프랭클린
#2015  3.0  2.0   1.0
#2016  2.0  3.0   1.0
#2017  1.0  3.0   2.0

df.columns
x = df.rank(ascending = True, axis = 1)
x
plt.plot(x)#x가 년도를 의도했는데 수열처럼 숫자가 나와있다.
##
plt.plot(x.iloc[:,0], label = '도바킨')
plt.plot(x.iloc[:,1], label = '게롤드', linestyle = '--')
plt.plot(x.iloc[:,2], label = '프랭클린', linestyle = ':')
plt.title('기록 순위 비교 그래프',fontsize=15)
plt.xlabel('년도',fontsize = 10)
plt.ylabel('순위',fontsize = 10)
plt.xticks(x.index, ['2015년','2016년','2017년'])
plt.yticks([1,2,3],['1등','2등','3등'])
plt.legend()#범례 

#emp에서 년도별 입사인원 보기
emp.info()
emp['HIRE_DATE']
#입사날짜 년도만 뽑아내자 
years = pd.to_datetime(emp['HIRE_DATE']).dt.year
years
type(years)#Series

x = years.value_counts()
type(x)
#년도 순서대로 정렬하고 그리기 
x = x.sort_index()
plt.plot(x)

#[문제 153] 2000 ~ 2016 년까지 성별 출생현황을 그래프로 생성하세요
#지난주에 했던 방식대로 데이터를 불러와 저장하자  
import glob
file = 'C:\\WorkSpace\\Python_Space\\csv\\yob*.csv'
file_lst = glob.glob(file)

df_yob = DataFrame()
year = 2000
year_birth = []
for i in file_lst:
    temp = DataFrame()
    temp = pd.read_csv(i, names = ['name','gender','number'])
    temp['year'] = year
    df_yob = df_yob.append(temp)
    year_birth.append([year, temp['number'].sum()])
    year += 1

df_yob
df_yob.info()
year_birth
#
x = df_yob.groupby(['year','gender'])['number'].sum().unstack()
type(x)
type(x.index)
#x의 index를 수정하자 
x.index = [str(i) + '년' for i in x.index]
##
plt.figure(figsize = (12,6))
plt.plot(x.iloc[:,0], label = '여성')
plt.plot(x.iloc[:,1], label = '남성')
plt.title('년도별 출생 그래프',fontsize=15)
plt.xlabel('년도',fontsize = 10)
plt.ylabel('출생',fontsize = 10)
#plt.yticks(range(0,2100000,100000))
plt.legend()


#지난주에 했던 137번 문제를 다시 보고 해보자
import csv
import os

with open('C:\\WorkSpace\\Python_Space\\data\\year_gender_total.txt',
          'w',
          newline='',
          encoding='utf-8') as f:
    #newline 없으면 빈 줄이 있는 상태로 저장된다.
    writer = csv.writer(f, delimiter = ',')#열을 ,로 나눈다.
    writer.writerow(['년도','여자','남자'])#행을 입력한다.
    for y in range(2000,2017):
        filename = 'C:\\WorkSpace\\Python_Space\\csv\\yob%d.csv'%y
        name = os.path.basename(filename)
        name = name.split('.')[0]
        df = pd.read_csv(filename, names = ['name','gender','birth'])
        gender_cn = df['birth'].groupby(df['gender']).sum()
        writer.writerow([name[3:], gender_cn.loc['F'], gender_cn.loc['M']])

df = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\year_gender_total.txt')
df.info() 
df
#
plt.figure(figsize = (12,6))
plt.plot(df.iloc[:,1], label = '여성')
plt.plot(df.iloc[:,2], label = '남성')
plt.title('년도별 출생 그래프',fontsize=15)
plt.xlabel('년도',fontsize = 10)
plt.ylabel('출생',fontsize = 10)
plt.xticks(df.index, df.iloc[:,0])
plt.legend()

##
#나라지표에서 데이터를 가져옴 
data = pd.read_excel('C:\\WorkSpace\\Python_Space\\data\\gdp.xls')
data.info()#DataFrame
data.index#RangeIndex(start=0, stop=2, step=1)
data.columns
#Index(['Unnamed: 0', '2010', '2011', '2012', '2013', '2014', '2015', '2016',
#       '2017'],dtype='object')
data.iloc[0]
data.iloc[1]
data.iloc[:,0]
#우선 문자열안에 , 를 제거하자, replace는 문자에 사용하는 함수 
data.iloc[0] = [i.replace(',','') for i in data.iloc[0]]
data.iloc[0]
#문자를 숫자형으로 바꾸자 
data.iloc[0,1:] = [int(i) for i in data.iloc[0,1:]]
data.iloc[1,1:] = [float(i) for i in data.iloc[1,1:]]
#이제 그래프를 그려보자 
plt.figure(figsize = (12,6))
plt.plot(data.iloc[0,1:], label = data.iloc[0,0])
plt.title('년도별 gdp',fontsize=15)
plt.xlabel('년도',fontsize = 10)
plt.ylabel(data.iloc[0,0],fontsize = 10)
plt.legend()
##
plt.figure(figsize = (12,6))
plt.plot(data.iloc[1,1:], label = data.iloc[1,0])
plt.title('년도별 경제성장률',fontsize=15)
plt.xlabel('년도',fontsize = 10)
plt.ylabel(data.iloc[1,0],fontsize = 10)
plt.legend()
##
plt.plot(data.columns[1:], data.iloc[0,1:],
         color = 'b',marker='o', linestyle = 'solid')
plt.xlabel('년도',fontsize = 10)
plt.ylabel(data.iloc[0,0],fontsize = 10)
plt.legend()

#다음과 같이 한 번에 변형하는 pandas 함수가 있다.
data = pd.read_excel('C:\\WorkSpace\\Python_Space\\data\\gdp.xls') 
y = data.iloc[0].str.replace(',','')
type(y)#Series
y.iloc[1:].astype('int')

plt.plot(y[1:],label = data.iloc[0,0])
plt.title('년도별 gdp',fontsize=15)
plt.xlabel('년도',fontsize = 10)
plt.ylabel(data.iloc[0,0] +'(단위 10억원)',fontsize = 10)

#################################################################
#9/26#
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

#어제 했던거 다시 보면서, 그룹형 막대그래프를 그려보자   
data = {"도바킨":[15,13,11],"게롤드":[13,14,15],"프랭클린":[10,9,12]} 
data
df = DataFrame(data, index = [2015,2016,2017])
df
x = df.rank(ascending = True, axis = 1)

x.plot(kind = 'bar')
plt.title('기록 순위 비교 그래프',fontsize=15)
plt.xlabel('년도',fontsize = 10)
plt.ylabel('순위',fontsize = 10)
plt.xticks(range(3), ['2015년','2016년','2017년'])
plt.yticks([1,2,3],['1등','2등','3등'])
#그럼 연도별로 모두의 등수가 표시된다. 비교해서 보기 좋다.

#이제 어제 년도별로 남녀 출생수를 보왔던걸 그룹형 막대그래프로 그려보자 
#다시 전에 했던 방식대로 불러온다. 
import glob
file = 'C:\\WorkSpace\\Python_Space\\csv\\yob*.csv'
file_lst = glob.glob(file)

df_yob = DataFrame()
year = 2000
year_birth = []
for i in file_lst:
    temp = DataFrame()
    temp = pd.read_csv(i, names = ['name','gender','number'])
    temp['year'] = year
    df_yob = df_yob.append(temp)
    year_birth.append([year, temp['number'].sum()])
    year += 1

df_yob
df_yob.info()
year_birth

x = df_yob.groupby(['year','gender'])['number'].sum().unstack()
x
#x의 index를 수정하자 
x.index = [str(i) + '년' for i in x.index]
##
x.plot(kind = 'bar')
plt.title('년도별 출생 그래프',fontsize=15)
plt.xlabel('년도',fontsize = 10)
plt.ylabel('출생',fontsize = 10)
plt.legend()

#또는 
import csv
import os

with open('C:\\WorkSpace\\Python_Space\\data\\year_gender_total.txt',
          'w',
          newline='',
          encoding='utf-8') as f:
    #newline 없으면 빈 줄이 있는 상태로 저장된다.
    writer = csv.writer(f, delimiter = ',')#열을 ,로 나눈다.
    writer.writerow(['년도','여자','남자'])#행을 입력한다.
    for y in range(2000,2017):
        filename = 'C:\\WorkSpace\\Python_Space\\csv\\yob%d.csv'%y
        name = os.path.basename(filename)
        name = name.split('.')[0]
        df = pd.read_csv(filename, names = ['name','gender','birth'])
        gender_cn = df['birth'].groupby(df['gender']).sum()
        writer.writerow([name[3:], gender_cn.loc['F'], gender_cn.loc['M']])

df = pd.read_csv('C:\\WorkSpace\\Python_Space\\data\\year_gender_total.txt')
df.info() 
df
#
x.plot(kind = 'bar', color = ['r','g'])
plt.title('년도별 출생 그래프',fontsize=15)
plt.xlabel('년도',fontsize = 10)
plt.ylabel('출생',fontsize = 10)
plt.xticks(df.index, df.iloc[:,0])
plt.legend()

#
x.plot(kind = 'bar', stacked = True)
df.iloc[:,1:3].plot(kind = 'bar')
plt.xticks(range(17), [str(i) + '년' for i in range(2000,2017)])
plt.show()

###
#death,csv : 사망원인 질병 번호 
#csv 파일을 ndarray 로 불러들이고 싶다. 
death = np.loadtxt(fname = 'C:\\WorkSpace\\Python_Space\\csv\\death.csv',
                   delimiter = ',',#데이터 구분 기준 
                   dtype = np.int)#불러들인 데이터 타입
type(death)
death.size#130
#130명을 임의로 추출하여 사망원인을 10가지로 분류한 결과 
#1 : 감염성 질환
#2 : 각종 암
#3 : 순환기 질환
#4 : 호흡기 질환
#5 : 소화기 질환
#6 : 각종 사고사 
#7 : 비뇨기 질환
#8 : 정신병
#9 : 노환
#10 : 신경계 질환 

#빈도표를 만들어 보자 
table = pd.crosstab(index = death, #불러들일 값
                    colnames = ['질병'], #index 이름
                    columns = '도수') #빈도수 체크한 column 이름 
table
table.index
#index 이름들을 교체하자 
table.index = ['감염성 질환','각종 암','순환기 질환','호흡기 질환','소화기 질환',
               '각종 사고사 ','비뇨기 질환','정신병','노환','신경계 질환']
table.columns
#
table.plot(kind = 'bar')
#
table.plot(kind = 'barh')
plt.xlabel('사망건수',fontsize = 10)

#데이터와 구간을 직접 설정해서 막대그래프를 그려보자 
ages = [1,9,10,11,14,20,27,29,31,37,39,40,42,45,50,51]
bins = [0,10,20,30,40,50,60]

#ages들의 값들이 다음처럼 구간에 들어가서 막대그래프가 그려지게 해아한다.
#0 < ages <= 10 10 < ages <= 20 20 < ages <= 30 
#30 < ages <= 40 40 < ages <= 50 50 < ages <= 60

#cut(x(array), bins(int,) : , right)
pd.cut(ages, bins, right = True)
#[(0, 10], (0, 10], (0, 10], (10, 20], (10, 20], ..., (30, 40],
# (40, 50], (40, 50], (40, 50], (50, 60]]
#Length: 16
#Categories (6, interval[int64]): [(0, 10] < (10, 20] <
# (20, 30] < (30, 40] < (40, 50] < (50, 60]]

pd.cut(ages, bins, right = False)
#Categories 의 구간이 [,)로 바뀌게 된다. 

pd.cut(ages, bins = [50,60,70,80])
#bins 안에 있지않는 데이터들은 모두 NaN로 표시된다. 
#NaN로 나오는 데이터들은 나중에 막대그래프를 그릴때 나오지 않는다. 

#
age_cut = pd.cut(ages, bins, right = True)
age_cut.codes#ages의 원소들의 어느 bin에 있는지 보임, bins의 index 값으로 나옴 
#array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5], dtype=int8)
age_cut.categories
pd.value_counts(age_cut)
pd.value_counts(age_cut).sort_index()
#(0, 10]     3
#(10, 20]    3
#(20, 30]    2
#(30, 40]    4
#(40, 50]    3
#(50, 60]    1
#dtype: int64

type(ages)#list 
#hist : histogram 을 그리는 함수
#hist(x(array) : 입력값, bins : 구간들)
#hist의 return : n(구간 막대의 값), bins(array), patches(list)
plt.hist(ages, bins = 6, facecolor = 'blue', alpha = 0.5)
plt.hist(ages, bins = 'auto', facecolor = 'blue',alpha= 0.5)
plt.hist(ages, bins = [50,60,70,80], facecolor = 'blue', alpha = 0.5)

#weight 정보를 불러들이자 
weight = np.loadtxt('C:\\WorkSpace\\Python_Space\\data\\weight.txt')
weight
type(weight)#ndarray
weight.shape#(5,10) 형태를 바꾸어보자 
weight = weight.reshape((50,))
weight

ranges = [50,55,60,65,70,75,80,85,90]

plt.hist(weight, bins = 'auto')
plt.hist(weight, bins = ranges)

#
pd.cut(weight, ranges, right = True)

#
x = Series(weight).sort_values()
x
plt.figure(figsize = (12,6))
x.plot(kind = 'bar')

#히스토그램의 여려 변수의 값들을 확인하자 
n, bins, patches = plt.hist(
        ages, bins = 6, facecolor = 'blue', alpha = 0.5, rwidth = 0.9)
print(n)#[2. 3. 1. 3. 4. 3.]#구간당 수치
n.max()
print(bins)#구간의 경계값
#[ 1.          9.33333333 17.66666667 26.         34.33333333 
#42.66666667 51.        ]
for i in patches:
    print(i)
#Rectangle(xy=(1.41667, 0), width=7.5, height=2, angle=0)
#Rectangle(xy=(9.75, 0), width=7.5, height=3, angle=0)
#Rectangle(xy=(18.0833, 0), width=7.5, height=1, angle=0)
#Rectangle(xy=(26.4167, 0), width=7.5, height=3, angle=0)
#Rectangle(xy=(34.75, 0), width=7.5, height=4, angle=0)
#Rectangle(xy=(43.0833, 0), width=7.5, height=3, angle=0)
#    
n, bins, patches = plt.hist(
        weight, bins = ranges, facecolor = 'blue', alpha = 0.5, rwidth = 0.9)
print(n)
print(bins)
for i in patches:
    print(i)

##
plt.figure(figsize = (10,6))
n, bins, patches = plt.hist(
        ages, bins = 6, facecolor = 'blue', alpha = 0.9, rwidth = 0.9)
#막대기마다 위에 숫자를 출력하게 하자 
for i in range(0, len(n)):
    plt.text(x = (bins[i] + bins[i+1])/2 - 1, y = n[i] + 0.1,#문자가 위치할 좌표
             s = '{}'.format(n[i]),#넣을 문자 
             fontsize = 12, 
             color = 'red')
#x축에 막대기마다 구간값이 출력되게 하자 
plt.yticks([])
plt.xticks([(bins[i] + bins[i+1])/2 for i in range(0, len(bins) - 1)],
            ["{:.1f} ~ {:.1f}".format(bins[i], bins[i+1]) 
            for i in range(0, len(bins) - 1)])
y_min, y_max = plt.ylim()
plt.ylim(y_min, y_max + 0.5)
##
##
plt.figure(figsize = (10,6))
n, bins, patches = plt.hist(
        weight, bins = 6, facecolor = 'blue', alpha = 0.9, rwidth = 0.9)
#막대기마다 위에 숫자를 출력하게 하자 
for i in range(0, len(n)):
    plt.text(x = (bins[i] + bins[i+1])/2 - 1,
             y = n[i] + n.max()*0.02,
             s = '{}'.format(n[i]),#넣을 문자 
             fontsize = 12, 
             color = 'red')
#x축에 막대기마다 구간값이 출력되게 하자 
plt.yticks([])
plt.xticks([(bins[i] + bins[i+1])/2 for i in range(0, len(bins) - 1)],
            ["{:.1f} ~ {:.1f}".format(bins[i], bins[i+1]) 
            for i in range(0, len(bins) - 1)])
y_min, y_max = plt.ylim()
plt.ylim(y_min, y_max + 0.5)
##

Series(weight).describe()
Series(weight).plot.box()

plt.boxplot(weight, flierprops = 
            dict(marker = 'o',markersize = 10, markerfacecolor = 'r'))
plt.show()

quartile = np.percentile(weight, [0,25,50,75,100])#각 %에 해당되는 값이 나온다. 
quartile#array([52.  , 68.25, 74.5 , 79.  , 93.  ])

#사분위의 범위
iqr = quartile[3] - quartile[1]
iqr

#이상치를 찾는 기준
lf = quartile[1] - 1.5 * iqr
uf = quartile[3] + 1.5 * iqr
lf#52.125
uf#95.125
weight < lf
weight[weight < lf]#array([52.])
weight > uf
weight[weight > uf]#array([], dtype=float64)

##선생님의 풀이##
import math
import numpy as np
#데이터를 불러온다
weight = np.loadtxt('C:\\WorkSpace\\Python_Space\\data\\weight.txt')
weight.shape
weight = weight.reshape((50,))

#데이터를 정렬한다. 
weight = weight[np.argsort(weight)]
weight

m = np.median(weight)#중앙값(데이터의 갯수가 짝수이면 실제 데이터 값이 아니다.)
Q1 = np.percentile(weight, 25, axis=0)#제1사분위 
Q3 = np.percentile(weight, 75, axis=0)#제3사분위
IQR = Q3 - Q1

#이상치를 찾을 기준과 이상치들 
lf = weight[weight <= Q1 - 1.5 * IQR]#lower fence
uf = weight[weight >= Q3 + 1.5 * IQR]#upper fence 

#중앙값을 기준으로 해서 경계값들을 찾는다 
upper_wisker = np.median(weight) + 1.5 * IQR
lower_wisker = np.median(weight) - 1.5 * IQR
weight[(weight <= lower_wisker) & (weight > lf)]
weight[(weight >= upper_wisker)]#여기서는 uf가 없어서 & 안함 

#이제 박스 그래프에 해당 숫자가 나오게 하자 
plt.boxplot(weight)
plt.text(1.01,lf,#(x,y)좌표
         s='{}'.format(float(lf)))#해당 위치에 넣을 값 
plt.text(1.01,lower_wisker,
         s='{}'.format(float(weight[(weight <= lower_wisker) & (weight > lf)])))
plt.text(1.01,upper_wisker,
         s='{}'.format(float(weight[(weight >= upper_wisker)])))
plt.text(1.01,Q1,
         s='{}'.format(float(weight[math.ceil(len(weight)*0.25)])))
plt.text(1.01,Q3,
         s='{}'.format(float(weight[math.ceil(len(weight)*0.75)])))
plt.text(1.01,m,
         s='{}'.format(float(np.median(weight))))
