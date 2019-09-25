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

###Numpy###
#-과학 계산을 위한 라이브러리로 다차원배열을 처리하는데 필요한 기능을 제공한다.
import numpy as np

z1 = np.array([1,2,3])
z1#array([1, 2, 3])
type(z1)#numpy.ndarray
z1.dtype#dtype('int32')

z2 = np.array([[1,2,3,4],[4,2,6,4]])
z2
#array([[1, 2, 3, 4],
#       [4, 2, 6, 4]])
type(z2)
z2.dtype
z2.shape#(2,4)

#list로 ndarray를 만들자 
lst = [[2,5,3],[4,7,6],[8,54,7]]
type(lst)
z3 = np.array(lst)
z3
#array([[ 2,  5,  3],
#       [ 4,  7,  6],
#       [ 8, 54,  7]])
z3.shape#(3,3)
#z3[행,열]
z3[0]
z3[1]
z3[2]

z3[:,0]
z3[:,1]
z3[:,2]
z3[0:2,0]

z3[1:,1:]
z3[0:2,0:2]
z3[0,0]
z3[1,1]
z3[2,2]

#
lst = [[1,2,3],[4,5,6],[7,8,9]]
n = np.array(lst)
n
b = np.array([[False, True, False],[True, False, True],[False, True, False]])
type(b)
b.dtype #dtype('bool')
b.shape

#True인 index만 골라내기 
n[b] #array([2, 4, 6, 8])
n[n%2 == 0] #array([2, 4, 6, 8])

#변수에 True/False 넣어서 하기 
r = n%2 == 0
n[r] #array([2, 4, 6, 8])

#ndarray 에 자동으로 숫자넣어서 만들기
np.zeros((3,3))
#array([[0., 0., 0.],
#       [0., 0., 0.],
#       [0., 0., 0.]])
np.ones((3,3))
#array([[1., 1., 1.],
#       [1., 1., 1.],
#       [1., 1., 1.]])
np.full((4,4),2)
#array([[2, 2, 2, 2],
#       [2, 2, 2, 2],
#       [2, 2, 2, 2],
#       [2, 2, 2, 2]])
np.eye(3)
#array([[1., 0., 0.],
#       [0., 1., 0.],
#       [0., 0., 1.]])
np.eye(4)
np.eye(4)

list(range(20))
z = np.array(range(20))
z.shape
#1차원 배열이니까 2차원 배열로 만들어 보자 
z = z.reshape((4,5))
z
z = z.reshape((20,))
z

#1차원 배열들의 연산
x = np.array([1,2,3])
y = np.array([4,5,6])

x[0] + y[0]
x[1] + y[1]
x[2] + y[2]
#같은 index끼리 더해진다.
x + y#array([5, 7, 9])
np.add(x,y)#array([5, 7, 9])

x - y#array([-3, -3, -3])
np.subtract(x,y)#array([-3, -3, -3])

x * y#array([ 4, 10, 18])
np.multiply(x,y)#array([ 4, 10, 18])

x / y#array([0.25, 0.4 , 0.5 ])
np.divide(x,y)#array([0.25, 0.4 , 0.5 ])

#2차원 배열들의 연산
lst1 = [[1,2],[3,4]]
lst2 = [[5,6],[7,8]]

x = np.array(lst1)
y = np.array(lst2)

x.shape
y.shape

x[0,0] + y[0,0]
x[0][0] + y[0][0]
x[0,0] - y[0,0]

x + y
np.add(x,y)

x - y
np.subtract(x,y)

x * y
np.multiply(x,y)

x / y
np.divide(x,y)

#행렬의 곱
np.dot(x,y)

x = np.array([[1,2],[3,4]])
x
#array([[1, 2],
#       [3, 4]])
#x안의 원소들을 모두 더하고 싶을 때 
np.sum(x)#10

np.sum(x, axis = 0)#열 기준 합 array([4, 6])
np.sum(x, axis = 1)#행 기준 합 array([3, 7])
np.mean(x)#2.5
np.var(x)#1.25
np.var(x, axis = 0)
np.var(x, axis = 1)
np.std(x)
np.std(x, axis = 0)
np.std(x, axis = 1)
np.max(x)
np.max(x, axis = 0)
np.max(x, axis = 1)
np.min(x)
np.min(x ,axis = 0)
np.min(x, axis = 1)

#모든 원소들의 최대값과 최소값을 구해보자 
x = np.array([[1,2],[3,0],[5,6]])
x
#array([[1, 2],
#       [3, 0],
#       [5, 6]])
x.shape
np.argmin(x)#index 번호가 나옴 
np.argmin(x.reshape((6,)))
np.argmin(x, axis = 0)#열마다 최소값 #array([0, 1], dtype=int64)
np.argmin(x, axis = 1)#행마다 최소값 #array([0, 1, 0], dtype=int64)

np.argmax(x) 
np.argmax(x.reshape((6,)))
np.argmax(x, axis = 0)#array([2, 2], dtype=int64)
np.argmax(x, axis = 1)#array([1, 0, 1], dtype=int64)

#누적합 구하기 
np.cumsum(x.reshape((6,)))
np.cumsum(x)
np.cumsum(x, axis = 0)
#array([[1, 2],
#       [4, 2],
#       [9, 8]], dtype=int32)
np.cumsum(x, axis = 1)
#array([[ 1,  3],
#       [ 3,  3],
#       [ 5, 11]], dtype=int32)

#누적곱 구하기 
np.cumprod(x.reshape((6,)))
np.cumprod(x)
np.cumprod(x, axis = 0)
#array([[ 1,  2],
#       [ 3,  0],
#       [15,  0]], dtype=int32)
np.cumprod(x, axis = 1)
#array([[ 1,  2],
#       [ 3,  0],
#       [ 5, 30]], dtype=int32)

np.prod(x)
np.prod(x, axis = 0)#array([15,  0])
np.prod(x, axis = 1)#array([ 2,  0, 30])

x = np.arange(5)
x.dtype
type(x)
#int32
2**31#2147483648
#int64
2**63#9223372036854775808

#기본값이 int32로 되어서 만들어진다. float형식으로 만들어 보자 
f = np.array(5, dtype = 'f')
f.dtype
type(f)

x = np.arange(3,10,dtype='f')
x
type(x)
x.dtype

x = np.arange(10)
x.shape
x.reshape((5,2))
x.reshape((5,2), order = 'C') #행우선으로 값이 채워진다. 기본값
#array([[0, 1],
#       [2, 3],
#       [4, 5],
#       [6, 7],
#       [8, 9]])
x.reshape((5,2), order = 'F') #열우선으로 값이 채워진다. 
#array([[0, 5],
#       [1, 6],
#       [2, 7],
#       [3, 8],
#       [4, 9]])

x = np.arange(10).reshape((5,2), order = 'F')
x
#array([[0, 5],
#       [1, 6],
#       [2, 7],
#       [3, 8],
#       [4, 9]])
x.reshape((10,), order = 'C')
x.flatten()

x.reshape((10,), order = 'F')
x.flatten('f')

x.ravel()
x.ravel('C')
x.ravel('F')

#
x = np.array([[5,7,22],[6,54,2]])
y = np.array([[8,77,12],[4,66,3]])
x
y
np.concatenate([x,y], axis = 0)
#array([[ 5,  7, 22],
#       [ 6, 54,  2],
#       [ 8, 77, 12],
#       [ 4, 66,  3]])
np.concatenate([x,y], axis = 1)
#array([[ 5,  7, 22,  8, 77, 12],
#       [ 6, 54,  2,  4, 66,  3]])
np.vstack((x,y))
#array([[ 5,  7, 22],
#       [ 6, 54,  2],
#       [ 8, 77, 12],
#       [ 4, 66,  3]])
#np.vstack((x,y), axis = 1)

#################################################################
#9/25#
#[문제 150] 원소의 값은 1 ~ 12 까지 행우선으로 3행4열 배열을 생성하세요
import pandas as pd
import numpy as np
x = np.array(range(12)).reshape((3,4), order = 'C')+1
x
y = np.array(range(1,13)).reshape((3,4), order = 'C')
y
y.shape
y.size # 행렬의 원소 개수
y.ndim # 차원의 수 
y.itemsize#원소 하나가 차지하는 바이트 값 
y.nbytes#배열 전체가 차지하는 바이트 값

vector_row = np.array([1,2,3])
vector_row.shape

vector_col = np.array([[1],[2],[3]])
vector_col.shape

x = np.arange(3)
x.repeat(2)#2번씩 반복
#array([0, 0, 1, 1, 2, 2])
x.repeat([2,3,4])#0번 원소는 2번 반복, 1번 원소는 3번 반복, 2번 원소는 4번 반복 
#array([0, 0, 1, 1, 1, 2, 2, 2, 2])

z = np.array([[1,2],[3,4]])
z.shape
z.repeat(2)
z.repeat(2, axis = 0)#열방향 반복
z.repeat(2, axis = 1)#행방향 반복

np.tile(z,2)
np.tile(x,2)

lst = [10,20,10,5,4,40,60,80,5,20,110,30]
from pandas import Series, DataFrame
Series(lst).unique()
Series(lst).nunique()#유일한 원소들의 갯수 
Series(lst).duplicated()#중복된 원소(앞에서 부터 해야려서 )

np.unique(lst)

lst = ['a','a','b','c','b','c']
np.unique(lst)
np.unique(lst, return_counts = True)
index, cn = np.unique(lst, return_counts = True)
print(index, cn)

u = np.array([[1,0,0],[1,0,0],[1,0,0]])
u
np.unique(u)#array([0, 1])
np.unique(u, axis = 0)#유일한 행
#array([[1, 0, 0]])
np.unique(u, axis = 1)#유일한 열 
#array([[0, 1],
#       [0, 1],
#       [0, 1]])

w = np.array([[1,1,1,1],[1,1,2,2],[1,2,2,2],[1,2,2,2]])
w
#array([[1, 1, 1, 1],
#       [1, 1, 2, 2],
#       [1, 2, 2, 2],
#       [1, 2, 2, 2]])
np.unique(w)
np.unique(w, axis = 0)#
#array([[1, 1, 1, 1],
#       [1, 1, 2, 2],
#       [1, 2, 2, 2]])
np.unique(w, axis = 1)
#array([[1, 1, 1],
#       [1, 1, 2],
#       [1, 2, 2],
#       [1, 2, 2]])

x = np.arange(0, 20, 2)
y = np.arange(0, 30, 3)
x
y
np.maximum(x,y)#둘중 큰수
np.minimum(x,y)#둘중 작은수

np.union1d(x,y)#합집합
np.intersect1d(x,y)#교집합
np.setdiff1d(x,y)#차집합

x = np.array([50,30,40,10,20])
x[:]
x[::]
x[:-1]
x[::-1]#역순으로 나열된다.
x.argsort()#정렬한 index 
x[x.argsort()]#오름차순
x[x.argsort()][::-1]#내림차순 
x[x.argsort()[::-1]]

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
#부서별 인원 
y = emp['DEPARTMENT_ID'].value_counts()

plt.figure(figsize = (10,10))
plt.pie(y, labels = y.index, colors = colors, autopct = '%1.1f%%',
        textprops = {'fontsize':20})
plt.title("부서별 인원 현황",fontsize = 20)
plt.show()

#직업별 인원
y = emp['JOB_ID'].value_counts()

plt.figure(figsize = (10,10))
plt.pie(y, labels = y.index, colors = colors, autopct = '%1.1f%%',
        textprops = {'fontsize':20})
plt.title("직업별 인원 현황",fontsize = 20)
plt.show()

#부서별 인원 
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

##막대그래프를 만들어 보자 
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
plt.xticks(x,xlabel)
plt.xlabel('과목', size = 15)
plt.ylabel('점수', size = 15)
plt.grid(True)
plt.show()

#막대가 누워서 만들어진다. 
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
plt.plot([0,5,10,15,20,25,30,20,5])
plt.plot([100,120,150,500,600,800],[1,5,7,9,15,33],
         color = 'y') #r,g,b,c,m,y,k,w 등의 약자들이 있다. 
#r = red, g = green, b = blue, c = cyan, m = magenta, y = yellow,
#k = black, w = white
plt.plot([100,120,150,500,600,800],[1,5,7,9,15,33],
         color = 'k')
plt.plot([100,120,150,500,600,800],[1,5,7,9,15,33],
         color = '0.75')
plt.plot([100,120,150,500,600,800],[1,5,7,9,15,33],
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
df.rank()
df.rank(axis = 0)#열별 순위, 기본값
df.rank(axis = 1)#행별 순위 

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
years = pd.to_datetime(emp['HIRE_DATE']).dt.year
years
type(years)#Series

x = years.value_counts()
type(x)
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