# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:18:52 2019

@author: stu11
"""

#[문제 162] 중앙일보에서 인공지능 관련 기사 스크랩핑을 하세요 
#current_url : url을 return 하는 것 
#res = req.urlopen(driver.current_url)
import urllib.request as req
from bs4 import BeautifulSoup

from selenium import webdriver
import time

url = "https://search.joins.com/?keyword"
driver = webdriver.Chrome("C:\WorkSpace//chromedriver.exe")
driver.get(url)

ser = driver.find_element_by_id("searchKeyword")
ser.send_keys("인공지능")
ser2 = driver.find_element_by_id("btnSearch")
ser2.click()
time.sleep(5)

#뉴스 더 보기 
'//*[@id="searchNewsArea"]/div[3]/a'
news_more = driver.find_element_by_xpath('//*[@id="searchNewsArea"]/div[3]/a')
news_more.click()
time.sleep(5)

#중앙일보 더 보기 
'//*[@id="content"]/div[2]/div[3]/a'
news_more = driver.find_element_by_xpath('//*[@id="content"]/div[2]/div[3]/a')
news_more.click()
time.sleep(5)

#이제 뉴스 검색 결과 리스트가 뜬다. 뉴스 주소를 저장하자 
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

list_all = soup.select("h2.headline.mg")
list_all

news_link = []
for i in list_all:
    news_link.append(i.select_one('h2 > a')['href'])
news_link

#링크 마다 들어가서 스크래핑하고...

news = []
for i in news_link:
    html_temp = req.urlopen(i)
    soup_temp = BeautifulSoup(html_temp, "html.parser")
    news.append(soup_temp.select_one('div#article_body').get_text(strip = True))

news
news[0].find('Copyright')#-1

###이제 위 과정을 페이지 목록을 눌러 들어가서 뉴스 주소를 모은거 

news_links = []
for num in range(2,11):
    res = req.urlopen(driver.current_url)
    soup = BeautifulSoup(html, "html.parser")
    list_all = soup.select("h2.headline.mg")
    
    for i in list_all:
        news_links.append(i.select_one('h2 > a')['href'])
    
    driver.implicitly_wait(3)
    driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div[3]/div/a[{}]'.format(num)).click()
    driver.implicitly_wait(3)

news2 = []
for i in news_links:
    html_temp = req.urlopen(i)
    soup_temp = BeautifulSoup(html_temp, "html.parser")
    news2.append(soup_temp.select_one('div#article_body').get_text(strip = True))

news2

news2[0]

driver.close()
########################
##선생님의 풀이##
import urllib.request as req
from bs4 import BeautifulSoup
from selenium import webdriver

driver = webdriver.Chrome("c:/data/chromedriver.exe")
driver.implicitly_wait(3)
driver.get('https://search.joins.com/')
keyword = "인공지능"
inputid = driver.find_element_by_id("searchKeyword")
inputid.clear()
inputid.send_keys(keyword)
driver.implicitly_wait(2)
driver.find_element_by_xpath('//*[@id="btnSearch"]').click()
driver.implicitly_wait(2)
driver.find_element_by_xpath('//*[@id="searchNewsArea"]/div[3]/a').click()
driver.implicitly_wait(2)
driver.find_element_by_xpath('//*[@id="content"]/div[2]/div[3]/a').click()
driver.implicitly_wait(2)

url = []
for no in range(2,11):
    res = req.urlopen(driver.current_url)
    soup = BeautifulSoup(res,"html.parser")
    for i in soup.select("ul.list_default > li > span.thumb"):
        url.append(i.select_one('a')['href'])
        
    driver.implicitly_wait(3)
    driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div[3]/div/a[{}]'.format(no)).click()
    driver.implicitly_wait(3)
    

news = []
for i in url:
    html = req.urlopen(i)
    soup= BeautifulSoup(html, "html.parser")
    news.append(soup.find('div',id="article_body").get_text(strip=True))

driver.close()
#####

#앞으로 배울 내용들 
#분류
#군집
#예측 
#등을 하려면 데이터가 필요하다. 정확한 데이터가 필요하다. 

#[문제 163] add함수를 생성하세요

result = 0
def add(x):
    global result
    result = result + x

add(20)
add(30)
result

#그런데 이러면 새롭게 계산을 할 수 없다.
#객체지향언어, 클래스를 배워보자 
#절차(구조적) 프로그램(procedural language) : c언어, plsql, R
#물이 위에서 아래로 흐르는 것처럼 순차적인 처리가 중요시 되며 
#프로그램 전체가 유기적으로 연결되도록 만드는 프로그래밍 기법
#단점 : 재사용할 수가 없다. 확장성이 떨어진다. 유지보수가 어렵다. 
#객체지향프로그램(object oriented language) : java, c++, c#, pathon
#구조적프로그래밍과 다르게 큰 문제를 작은문제들로 해결할 수 있는
#객체들을 만든뒤 이 객체들을 조합해서 큰 문제를 해결하는 방법
#객체 : 사물 개념중에서 명사로 표현할 수 있는것들을 의미한다. 
#사람, 건물, 학생 등등
#클래스 : 객체를 설명해 놓은 것(설계도)
#인스턴스 : 클래스를 메모리에 만들어서 사용하도록 하는 의미 

#객체 = 속성 + 메소드 
#객체 : 사람 
#속성(변수) : 팔, 다리, 머리, 눈, 코, 입, 이름, 키, 나이, 주소 
#메소드(함수) : 속성의 값을 변경, 동작하게 하는 프로그램 

class Calculator:
    def __init__(self):
        self.result = 0
    def add(self, num):
        self.result += num
        return self.result

#클래스의 인스턴스 화
cal1 = Calculator()
cal1.add(10)
cal1.add(2)
cal1.add(9)
cal1.add(10)
#31

cal2 = Calculator()
cal2.add(8)
cal2.add(2)
cal2.add(8)
cal2.add(9)
#27

# __init__ : 클래스를 인스턴스 할때 자동으로 생성자가 실행,
#인스턴스의 초기상태/값등을 지정하는 메소드
#self : 자기 자신의 클래스를 의미 (java에서는 dis 라고 한다. )

class Person:
    name = "홍길동"
    age = 20
    #class 안에 함수에 넣을게 없어도 self 넣어야 한다. 
    def myPrint (self):
        print("이름은 {}".format(self.name))
        print("나이는 {}".format(self.age))
        
p1 = Person()
p1.myPrint()
p1.name
p1.age

##self 없이 만들어 보자 
class Person2:
    name = "홍길동"
    age = 20
    def myPrint ():
        print("이름은 {}".format(name))
        print("나이는 {}".format(age))

p2 = Person2()
p2.myPrint()#error 
p2.name 
p2.age

##
p3 = Person()
p3.name = "dovahkiin"
p3.name#'dovahkiin' , name 값이 바뀐다.
p1.name
p3.myPrint()

#원래 class에 없는 값을 입력해보자 
p3.job = "Last Dragonborn"
print(p3.name,"직업은",p3.job)

##이름과 나이를 저장할 수 있는 클래스를 만들자 
class Person3:
    info = ""
    def showinfo(self, name, age):
        self.info += "이름 : " + name + " 나이 : " + str(age) + "\n"
      
man = Person3()
man.showinfo('dovahkiin',22)
man.showinfo('hanzo',40)
print(man.info)

woman = Person3()
woman.showinfo('한가인',60)
woman.showinfo('김태희',50)
print(woman.info)

#class에 초기화 기능을 추가해 보자 
class Person4:
    def __init__(self):        
        self.info = ""
    def showinfo(self, name, age):
        self.info += "이름 : " + name + " 나이 : " + str(age) + "\n"
        
man = Person4()
man.showinfo('dovahkiin',22)
man.showinfo('hanzo',40)
print(man.info)

woman = Person4()
woman.showinfo('한가인',60)
woman.showinfo('김태희',50)
print(woman.info)
#지금은 Person3과 Person4가 차이점이 없다.

class Happy:
    def __init__(self):
        print("오늘 하루도 행복하세요")
        
happy = Happy()

class Happy2:
        print("오늘 하루도 행복하세요")
        
happy = Happy2()
#결과가 같다

class Employee:
    #인스턴스변수, 자신의 인스턴스 안에서만 글로벌 변수이다. 
    empCount = 0
    
    def __init__(self, arg1, arg2):
        self.name = arg1
        self.sal = arg2
        self.empCount += 1
        
    def displayCount(self):
        print("전체 종업원 수는 %d"%self.empCount)
        
    def displayEmployee(self):
        print("이름 : ", self.name, "급여 : ", self.sal)
  
#emp = Employee() error, arg값을 넣어야 한다 
emp = Employee('겐지',2000)
emp.displayCount()
emp.displayEmployee()
emp.name
emp.sal

emp.name = '로드호그'
emp.sal = 6000
emp.displayCount()#변하지 않는다. 
emp.displayEmployee()

emp.empCount

emp2 = Employee('리퍼',3000)
emp2.displayCount()
emp2.displayEmployee()

emp.__init__('dovahkiin',10000)
emp.displayCount()
emp.displayEmployee()

#다음과 같이 바꾸어 보자 
class Employee:
    empCount = 0
    
    def __init__(self, arg1, arg2):
        self.name = arg1
        self.sal = arg2
        #바꿈, 이것은 진정한 글로별 변수가 된다. 
        Employee.empCount += 1
        
    def displayCount(self):
        print("전체 종업원 수는 %d"%self.empCount)
        
    def displayEmployee(self):
        print("이름 : ", self.name, "급여 : ", self.sal)
        
emp = Employee('겐지',2000)
emp.displayCount()
emp.displayEmployee()

emp2 = Employee('리퍼',3000)
emp2.displayCount()#숫자가 증가한다. !!!
emp2.displayEmployee()

#[문제 163] 초기 생성자에는  이름, 주소, 급여를 입력값으로 받고 
#아래와 같이 출력되는....
class Employee:
    #instance 갯수
    instanceCount = 0
    #등록 수 
    empCount = 0
    
    def __init__(self):
        self.info = ""
        #instance에 저장된 인원 수를 나타내게 된다. 
        self.cn = 0
        #이 class에서 만들어진 instance의 갯수를 나타내게 된다.
        Employee.instanceCount += 1
        
    def ClassCount(self):
        print("전체 class 수는 %d"%Employee.instanceCount)
        
    def addEmployee(self, name, addr, sal):
        self.info += "이름 : " + name.rjust(8) +\
        ", 주소 : " +addr.rjust(8)+\
        ", 급여 : " + str(sal).rjust(8) + "\n"
        self.cn += 1
        self.empCount += 1
        Employee.empCount += 1
    
    def printEmployee(self):
        print(self.info)
        print("instance 안에 인원수",self.cn)
        print(self.empCount)
        print("모든 instance 안에 인원수는 %d"%Employee.empCount)

emp1 = Employee()
emp1
emp1.addEmployee("고길동","도봉구",5000)
emp1.addEmployee("둘리",'부천시',20000)
emp1.printEmployee()
emp1.ClassCount()

emp2 = Employee()
emp2.addEmployee("토르비욘","포탑",10000)
emp2.printEmployee()
emp2.ClassCount()

emp3 = Employee()
emp3.addEmployee("겐지","내가 용이 된다",7600)
emp3.addEmployee("dovahkiin","fus ro dah",4555)
emp3.printEmployee()
emp3.ClassCount()

####선생님의 풀이 ####
class Employee:
   empCn = 0
   def __init__(self):
      Employee.empCn += 1
      self.info = ""
      self.cn = 0
   
   def printCount(self):
       print("클래스 수 : %d" %Employee.empCn)
   def addEmployee(self, name, addr, salary):
       self.info += "이름 : "+name+" 주소 : "+addr+ " 급여 : " + str(salary)+"\n"
       self.cn += 1
   def printEmployee(self):
       print(self.info)
       print("인스턴스안에  인원수",self.cn)

emp1 = Employee()
emp1.addEmployee('홍길동','덴마크',1000)
emp1.addEmployee('박찬호','핀란드',2000)
emp1.printCount()
emp1.printEmployee()


emp2 = Employee()
emp2.addEmployee("홍아들","노르웨이", 2000)
emp2.addEmployee("손흥민","영국", 3000)
emp2.addEmployee("류현진","미국", 5000)
emp2.printCount()
emp2.printEmployee()

####################################################
class Employee:
    
    empCn = 0
    def __init__(self):
        
        self.info = ""
    def printCount(self):
        print("전체 인원수 : %d" %Employee.empCn)
    def addEmployee(self, name, addr, salary):
        self.info += "이름 : "+name+" 주소 : "+addr+ " 급여 : " + str(salary)+"\n"                            
        Employee.empCn += 1
    def printEmployee(self):
        print(self.info)
     
emp1 = Employee()
emp1.addEmployee('홍길동','덴마크',1000)
emp1.addEmployee('박찬호','핀란드',2000)
emp1.printCount()
emp1.printEmployee()


emp2 = Employee()
emp2.addEmployee("홍아들","노르웨이", 2000)
emp2.addEmployee("손흥민","영국", 3000)
emp2.addEmployee("류현진","미국", 5000)
emp2.printCount()
emp2.printEmployee()

#########################################
class Employee:
    
    empCn = 0
    info = ""
    def printCount(self):
        print("전체 인원수 : %d" %Employee.empCn)
    def addEmployee(self, name, addr, salary):
        #다른 인스턴스에서 정보를 입력해도 모두 합쳐진다. 
        Employee.info += "이름 : "+name+" 주소 : "+addr+ " 급여 : " + str(salary)+"\n"                            
        Employee.empCn += 1
    def printEmployee(self):
        print(Employee.info)
     
emp1 = Employee()
emp1.addEmployee('홍길동','덴마크',1000)
emp1.addEmployee('박찬호','핀란드',2000)
emp1.printCount()
emp1.printEmployee()


emp2 = Employee()
emp2.addEmployee("홍아들","노르웨이", 2000)
emp2.addEmployee("손흥민","영국", 3000)
emp2.addEmployee("류현진","미국", 5000)
emp2.printCount()
emp2.printEmployee()

#########################################
class Employee:
    
    empCn = 0
    def __init__(self):
        #이러면 나중에 새로운 인스턴스를 생성할때 다른 인스턴스에서 입력한 
        #데이터들이 사라진다. 
        Employee.info = ""
    def printCount(self):
        print("전체 인원수 : %d" %Employee.empCn)
    def addEmployee(self, name, addr, salary):
        #다른 인스턴스에서 정보를 입력해도 모두 합쳐진다. 
        Employee.info += "이름 : "+name+" 주소 : "+addr+ " 급여 : " + str(salary)+"\n"                            
        Employee.empCn += 1
    def printEmployee(self):
        print(Employee.info)
     
emp1 = Employee()
emp1.addEmployee('홍길동','덴마크',1000)
emp1.addEmployee('박찬호','핀란드',2000)
emp1.printCount()
emp1.printEmployee()

emp2 = Employee()
emp2.addEmployee("홍아들","노르웨이", 2000)
emp2.addEmployee("손흥민","영국", 3000)
emp2.addEmployee("류현진","미국", 5000)
emp2.printCount()
emp2.printEmployee()
####################################################################
#10/8#

class Person:
    #클래스 변수 
    hobbys = []
    
    def __init__(self,hobby):
        #hobbys가 클래스 변수이다. 모든 인스턴스에서 사용이 가능하다.
        self.hobbys.append(hobby)

p1 = Person("독서")
p1.hobbys

p2 = Person("게임")
p2.hobbys
#['독서', '게임'], '독서','게임' 둘 다 추가되어 있다.

Person.hobbys
Person.hobbys.append('여행')
Person.hobbys
#['독서', '게임', '여행'], '여행'이 추가된다.

class Person:
    hobbys = []
    
    def add_hobby(self,hobby):
        self.hobbys.append(hobby)
        
p1 = Person()
p1.add_hobby("독서")
p1.hobbys

p2 = Person()
p2.add_hobby("게임")
p2.hobbys


class Person:
    def add_hobby(self, hobby):
        #인스턴스 변수 
        self.hobbys = hobby
    
    def show(self):
        print("내 취미는 ",self.hobbys)
        
p1 = Person()
p1.add_hobby("독서")
p1.show()#인스턴스 변수 
Person.hobbys#클래스 변수가 아니라서 오류

p2 = Person()
p2.add_hobby("게임")
p2.show()

#[문제 164] 주민번호를 입력하면 다음과 같이 출력되는 클래스를 생성하시오

#id1 = Processldc("010101-3234567")
#id1.id_process()
#('2001','01','01','남성')
#id2 = Processldc("990202-2123456")
#id2.id_process()
#('2001','01','01','여성')

import re
bool(re.match('\d{6}\-\d{7}','010101-3234567'))
str1 = '010101-3234567'
str1[7]

class PersonNumber:
    
    
    def __init__(self,x):
        
        self.data = []
        #self.deter = 0
        
        #외국인은 -뒤의 번호가 1~4가 아니라고 한다. 여기서는 넘어간다. 
        if not bool(re.match('\d{2}[0|1]\d[0-3]\d\-[1-4]\d{6}',x)):
            print('초기화 실패')
        else:
            self.personNumber = x
            
    def NumberPrint(self):
        print(self.personNumber)
        
    def id_process(self):
        deter = int(self.personNumber[7])
                     
        #출생 세기에 따라서 
        if deter <=2:
            self.data.append('19'+self.personNumber[0:2])
        else:
            self.data.append('20'+self.personNumber[0:2])
            
        self.data.append(self.personNumber[2:4])
        self.data.append(self.personNumber[4:6])
           
        #성별에 따라서 
        if deter in [1,3]:
            self.data.append("남성")
        else:
            self.data.append("여성")
            
        print(self.data)
        self.data = []

            
        
a1 = PersonNumber('123')       
a1.NumberPrint()

a2 = PersonNumber("010101-3234567")
a2.id_process() 
a2.NumberPrint()       

a3 = PersonNumber("990202-2123456")
a3.id_process() 

a4 = PersonNumber("950403-1893873")
a4.id_process()

a5 = PersonNumber("020202-4432343")
a5.id_process()

a6 = PersonNumber("020202-5432343")

##선생님 풀이 ##
class Processldc:
    def __init__(self,id):
        self.id = id
        
    def id_process(self):
        self.first, self.second = self.id.split('-')
        self.gender = self.second[0]
        
        if self.gender == '1' or self.gender == '2':
            self.year = '19'+self.first[:2]
        else:
            self.year = '20'+self.first[:2]
        
        if self.gender == "2" or self.gender == '4':
            self.gender = "여성"
        else:
            self.gender = '남성'
            
        self.month = self.first[2:4]
        self.day = self.first[4:6]
        
        return self.year, self.month, self.day, self.gender

a1 = Processldc('123')       
a1.NumberPrint()

a2 = Processldc("010101-3234567")
a2.id_process() 
a2.NumberPrint()       

a3 = Processldc("990202-2123456")
a3.id_process() 

a4 = Processldc("950403-1893873")
a4.id_process()

a5 = Processldc("020202-4432343")
a5.id_process()

a6 = Processldc("020202-5432343")   

#[문제 165] 위에 show()함수를 추가해 주세요 
#주민번호, 출생년월일, 성별, 나이를 출력하게 된다. 

class Processldc2:
    def __init__(self,id):
        self.id = id
        
    def id_process(self):
        self.first, self.second = self.id.split('-')
        self.gender = self.second[0]
        
        if self.gender == '1' or self.gender == '2':
            self.year = '19'+self.first[:2]
        else:
            self.year = '20'+self.first[:2]
        
        if self.gender == "2" or self.gender == '4':
            self.gender = "여성"
        else:
            self.gender = '남성'
            
        self.month = self.first[2:4]
        self.day = self.first[4:6]
        
        return self.year, self.month, self.day, self.gender
    
    def show(self):
        print("주민번호 : "+self.id)
        self.id_process()
        print("출생년월일 : "+self.year+"년 "+self.month+"월 "+self.day+"일")
        print("성별 : "+self.gender)
        import datetime
        d = datetime.datetime.now()
        old = (int(d.year) - int(self.year))+1
        print("나이 : "+str(old))

a2 = Processldc2("010101-3234567")
 
a2.show()      

a3 = Processldc2("990202-2123456")
a3.show()

a4 = Processldc2("950403-1893873")
a4.id_process()

a5 = Processldc2("020202-4432343")
a5.id_process()

a6 = Processldc2("020202-5432343") 

########   
class PersonNumber2:
    
    def __init__(self,x):
        
        self.data = []
        #외국인은 -뒤의 번호가 1~4가 아니라고 한다. 여기서는 넘어간다. 
        if not bool(re.match('\d{2}[0|1]\d[0-3]\d\-[1-4]\d{6}',x)):
            print('초기화 실패')
        else:
            self.personNumber = x
            
    def NumberPrint(self):
        print(self.personNumber)
        
    def id_process(self):
        self.deter = int(self.personNumber[7])
                     
        #출생 세기에 따라서 
        if self.deter <=2:
            self.year = '19'+self.personNumber[0:2]
        else:
            self.year = '20'+self.personNumber[0:2]
         
        self.data.append(self.year)
        self.data.append(self.personNumber[2:4])
        self.data.append(self.personNumber[4:6])
           
        #성별에 따라서 
        if self.deter in [1,3]:
            self.gender = "남성"
        else:
            self.gender = "여성"
         
        self.data.append(self.gender)
        return self.data
        self.data = []
        
    def show(self):
        print("주민번호 : "+ self.personNumber)
        self.id_process()
        print("출생년월일 : "+self.year + '년 '+
              self.personNumber[2:4] + '월'+self.personNumber[4:6]+'일')
        print("성별 : " + self.gender)
        import datetime
        d = datetime.datetime.now()
        old = (int(d.year) - int(self.year))+1
        print("나이 : "+str(old))
    
              
a1 = PersonNumber2('123')       
a1.NumberPrint()

a2 = PersonNumber2("010101-3234567")
a2.id_process() 
a2.NumberPrint()       

a3 = PersonNumber2("990202-2123456")
a3.show()

a4 = PersonNumber2("950403-1893873")
a4.show()

a5 = PersonNumber2("020202-4432343")
a5.show()

a6 = PersonNumber2("020202-5432343")

#나중에 나이계산을 만나이로 해보자, SQL로도 만들어보자 
#from datetime import date
import pandas as pd
pd.datetime.now() - pd.Timestamp('2000-01-01')
today = pd.datetime.now()
birth = pd.Timestamp('2000-01-01')
today.year - birth.year
today.month - birth.month
today.day - birth.day


#[문제 166] 
class Person:
    
    def __init__(self, name):
        self.name = name
        self.hobbys = ''
    
    def add_hobby(self, x):
        self.hobbys = self.hobbys + x + ' '
    
    def show(self):
        self.result = self.name + " 취미는 " + self.hobbys
        return self.result

p1 = Person("도바킨")
p1.hobbys
p1.add_hobby("게임하기")
p1.add_hobby("산책하기")
p1.hobbys
p1.show()

p2 = Person("게롤드")
p2.hobbys += "검술"
p2.show()

#[문제167] 아래와 같이 수행되는 클래스를 생성하세요.
#emp1 = Employee("홍길동",1000)
#emp1.emp_info()
#이름 :  홍길동 , 급여 :  1000

#emp1.raise_ratio
#1.1
#
#emp1.raise_salary()
#emp1.emp_info()
#이름 :  홍길동 , 급여 :  1100
#
#emp1.raise_ratio = 1.2
#
#emp1.raise_salary()
#
#emp1.emp_info()
#이름 :  홍길동 , 급여 :  1320
#
#emp2 = Employee("박찬호",2000)
#emp2.emp_info()
#이름 :  박찬호 , 급여 :  2000
#
#emp2.raise_ratio
#Out[26]: 1.1
class Employee:
    
    def __init__(self, name, sal):
        self.name = name
        self.sal = sal
        self.raise_ratio = 1.1
    
    def raise_salary(self):
        self.sal = self.sal * self.raise_ratio
    
    def emp_info(self):
        print("이름 : {}, 급여 : {}".format(self.name, self.sal))
   
emp1 = Employee('겐지',3000)
emp1.emp_info()
emp1.raise_ratio
emp1.raise_salary()
emp1.emp_info()

emp2 = Employee("리퍼",4500)
emp2.emp_info()
emp2.raise_ratio

#[문제 168] Stats 클래스를  정하세요

class Stats:
    
    def __init__(self):
        self.sum = 0
    
    def s_sum(self, *arg):
        self.sum = 0
        
        for i in arg:
            self.sum += i
        
        return self.sum
    
    def s_mean(self, *arg):
        #같은 클래스의 함수를 호출할때는 클래스이름.함수 
        self.mean = Stats.s_sum(self,*arg)/len(arg)
        
        return self.mean
    
    def s_variance(self, *arg):        
        self.variance = 0
        
        Stats.s_mean(self,*arg)
        
        for i in arg:           
            self.variance += (i - self.mean)**2
        
        self.variance /= (len(arg)-1)
        
        return self.variance
    
    def s_std(self, *arg):       
        Stats.s_variance(self,*arg)
        
        self.std = self.variance**(1/2)
        
        return self.std
            
stats = Stats()           
stats.s_sum(1,2,3,4,5)            
stats.s_mean(1,2,3,4,5,6)
stats.s_variance(1,2,3,4,5,6)


#상속
#클래스내에서 메소드 속성을 물려 받는다.
#공통된 내용/중복된 내용을 하나로 묶어서 관리할 수 있다. 
#

class Parents:
    
    def __init__(self, name, pn):
        self.name = name
        self.pn = pn
        
    def Printinfo(self):
        print("이름 : {}, 전화번호 : {}".format(self.name, self.pn))
        
p = Parents("고길동","000-0000-0000")
p.Printinfo()

#상속을 받는 class를 만들자 
class Child(Parents):
    
    def __init__(self, name, pn, addr, sn):
        self.name = name
        self.pn = pn
        self.addr = addr
        self.sn = sn
        
c = Child("홍하드","010-1004-1004","서울","000000-1234567")
#부모 class의 함수를 사용할 수 있다. 
c.Printinfo()

#부모 클래스에 있는 부분을 대체해 보자 
class Child2(Parents):
    
    def __init__(self, name, pn, addr, sn):
        Parents.__init__(self,name,pn)
        self.addr = addr
        self.sn = sn
    #그리고 새로운 함수를 추가해보자    
    def showInfo(self):
        print("주소 : {}, 주민번호 : {}".format(self.addr, self.sn))
        
c2 = Child2("홍하드","010-1004-1004","서울","000000-1234567")
c2.Printinfo()
c2.showInfo()

#[문제169] Person 클래스를 생성하세요. 생성자는 이름, 나이, 성별을 만드세요.
#Person 클래스 에는 printMe 메소드를 생성하셔서 이름, 나이 성별을 출력합니다.
#Employees클래스를 생성한후 Person상속받습니다.
#생성자는 이름, 나이, 성별, 주소, 생일입니다.
#단 이름, 나이, 성별은 person에서 상속받으세요.
#Employees 클래스에 printMe를 재구성하셔서 주소, 생일을 출력하세요.

class Person:
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender
        
    def printMe(self):
        print("이름 :{}, 나이 :{}, 성별 :{}"
              .format(self.name, self.age, self.gender))

p1 = Person("한조",40,"남성")
p1.printMe()

class Employees(Person):
    def __init__(self,name, age, gender, addr, birth):
        Person.__init__(self, name, age, gender)
        self.addr = addr
        self.birth = birth
        
    def printMe(self):
        Person.printMe(self)
        print("주소 : {}, 생일 : {}".format(self.addr, self.birth))
        
e1 = Employees("겐지",35,"남성","하나무라","20401001")
e1.printMe()

#[문제170] Add 클래스에 두수를 더하는 값을 리턴하는 add 메소드 생성
#Multiply 클래스에 두수를 곱한값을 리턴하는 multiply 메소드 생성
#Divide 클래스에 두수를 나눈값을 리턴하는 divide메소드 생성
#Calculator클래스에는 Add, Multiply, Divide 상속받고 두수를 뺀값을 리턴하는 sub 메소드 생성하세요.
#
#cal = Calculator()
#print(cal.add(10,20))
#print(cal.multiply(10,20))
#print(cal.divide(10,2))
#print(cal.sub(10,8))

class Add:

    def add(self,x,y):
        self.sum_num = x + y
        return self.sum_num

class Multiply:

    def multiply(self,x,y):
        self.mul_num = x * y
        return self.mul_num
    
class Divide:

    def divide(self,x,y):
        try:
            self.div_num = x / y
            return self.div_num
        except ArithmeticError as error:
            print(error)
        

class Calculator(Add, Multiply, Divide):

    def sub(self,x,y):
        self.sub_num = x - y
        return self.sub_num


cal = Calculator()
print(cal.add(10,20))
print(cal.multiply(10,20))
print(cal.divide(10,0))
print(cal.sub(10,8))        

#다만 다음과 같이 인스턴스를 만들지 않고 클래스의 함수를 바로 작동하면 에러가 난다. 
Calculator.add(1,2)
#static method를 만들면 된다. 이때는 self 지시어를 사용하면 안된다.
#@staticmethod : static method라는걸 나타냄 

class Calculator:
    @staticmethod
    def add(x,y):
        return x + y
    @staticmethod
    def multiply(x,y):
        return x * y
    @staticmethod
    def divide(x,y):
        try:
            return x/y
        except ArithmeticError as error:
            print(error)
    @staticmethod
    def sub(x,y):
        return x - y

Calculator.add(1,2)
Calculator.multiply(3,4)
Calculator.divide(4,3)
Calculator.sub(8,4)

##선생님이 올리신 파일을 불러들여 사용해 보자 
import sys
sys.path
#파일 경로 추가 
sys.path.append('C:\\Workspace\\Python_Space\\myPython')
from Stats_class import Stats as st
dir(st)

st.mean(1,2,3)
st.sum(1,2,3)
st.variance(1,2,3,4,5)
st.stddev(1,2,3,4)

########################################################################
#10/10#
#[문제 171] 초기 생성자에 이름, 핸드폰번호, 메일, 주소 변수를 생성합니다. 
#print_info 메소드를 생성한 후  출력하는 Contact 클래스를 생성하세요.
#인스턴스는 set_contact 함수를 이용해서 만드시고 이름, 핸드폰번호,메일,
#주소는 입력값으로 받아서 출력하세요.
#
#set_contact()
#
#이름을 입력하세요 : 홍길동
#
#핸드폰번호를 입력하세요 : 010-1000-1004
#
#메일을 입력하세요 : hong@aaa.com
#
#주소를 입력하세요 : 서울시 강남구 삼성로
#이름 : 홍길동 
#핸드폰번호 : 010-1000-1004 
#메일 : hong@aaa.com 
#주소 : 서울시 강남구 삼성로 

class Contact:
    
    def __init__(self, name, phone_number, mail, addr):
        self.name = name
        self.phone_number = phone_number
        self.mail = mail
        self.addr = addr
        
    def print_info(self):
        print("이름 : {}".format(self.name))
        print("핸드폰 번호 : {}".format(self.phone_number))
        print("메일 : {}".format(self.mail))
        print("주소 : {}".format(self.addr))

def set_contract():
    import re
    
    name = input("이름을 입력하세요 :")
   
    while(True):
        phone_number = input("핸드폰 번호를 입력하세요 :")
        if bool(re.match('\d{3}-\d{3,4}-\d{4}',phone_number)):
            break
        else:
           print("잘못된 형식입니다.") 
    
    while(True):
        mail = input("메일을 입력하세요 :")
        if bool(re.search('@',mail)):
            break
        else:
            print("잘못된 형식입니다.")
        
    addr = input("주소를 입력하세요 :")
    
    contact = Contact(name, phone_number, mail, addr)
    contact.print_info()
    
set_contract()

#이제 입력받은 값을 dbms에 저장해보자 
#[문제 172] Contact 클래스 이용해서 입력 들어온 값들을
#저장해보자 

#import sqlite3
#dir(sqlite3)
##디스크에 sqlite 구성하기 
#conn = sqlite3.connect("C:\\WorkSpace\\Python_Space\\data\\info.db")
##cursor 설정, 객체 지정 
#c = conn.cursor()

class Contact2:
       
    def __init__(self, name, phone_number, mail, addr):
        self.name = name
        self.phone_number = phone_number
        self.mail = mail
        self.addr = addr
        
    def print_info(self):
        print("이름 : {}".format(self.name))
        print("핸드폰 번호 : {}".format(self.phone_number))
        print("메일 : {}".format(self.mail))
        print("주소 : {}".format(self.addr))
        
    def create_table_db(self):
        import sqlite3
        conn = sqlite3.connect("C:\\WorkSpace\\Python_Space\\data\\info.db")
        c = conn.cursor()
        try:
            c.execute("create table person_inform\
                      (name varchar2(20),\
                       phone_number varchar2(20),\
                       mail varchar2(20),\
                       addr varchar2(20))")
        except:
            pass
        conn.commit()
        c.close()
        conn.close()
        
    def save_inform(self):
        import sqlite3
        conn = sqlite3.connect("C:\\WorkSpace\\Python_Space\\data\\info.db")
        c = conn.cursor()
        insert_sql = "insert into person_inform(name, phone_number, mail, addr)\
                      values (?,?,?,?)"
        c.execute(insert_sql,(self.name, self.phone_number, self.mail, self.addr))
        conn.commit()
        c.close()
        conn.close()
        
    def select_inform(self):
        import sqlite3
        conn = sqlite3.connect("C:\\WorkSpace\\Python_Space\\data\\info.db")
        c = conn.cursor()
        c.execute("select * from person_inform")
        c.fetchall()
        c.close()
        conn.close()

def set_contract2():
    import re
    
    name = input("이름을 입력하세요 :")
   
    while(True):
        phone_number = input("핸드폰 번호를 입력하세요 :")
        if bool(re.match('\d{3}-\d{3,4}-\d{4}',phone_number)):
            break
        else:
           print("잘못된 형식입니다.") 
    
    while(True):
        mail = input("메일을 입력하세요 :")
        if bool(re.search('@',mail)):
            break
        else:
            print("잘못된 형식입니다.")
        
    addr = input("주소를 입력하세요 :")
    
    contact2 = Contact2(name, phone_number, mail, addr)
    contact2.print_info()
    contact2.create_table_db()
    contact2.save_inform()
    
set_contract2()

import sqlite3
conn = sqlite3.connect("C:\\WorkSpace\\Python_Space\\data\\info.db")
c = conn.cursor()
c.execute("select * from person_inform")
c.fetchall()
c.close()
conn.close()

##선생님의 풀이 ##
class Contact11:
    def __init__(self,name, pn, email, addr):
        self.name = name
        self.pn = pn
        self.email = email
        self.addr = addr

    def print_info(self):
        print("이름 : {} ".format(self.name))
        print("핸드폰번호 : {} ".format(self.pn))
        print("메일 : {} ".format(self.email))
        print("주소 : {} ".format(self.addr))
    
    def open(self):
        import sqlite3
        self.conn = sqlite3.connect('c:/data/contact.db')
        self.c = self.conn.cursor()
        print("contact db open")
    
    def table_check(self):
        self.c.execute("SELECT name FROM sqlite_master WHERE name = 'contact'")
        if self.c.fetchone()  is None:
            self.c.execute('create table contact\
                           (name text,pn text, mail text,addr text)')
               
        
    def input(self):
        self.c.execute("insert into contact(name, pn, mail, addr)\
                       values(?,?,?,?)",(self.name,self.pn,self.email,self.addr))
        self.c.execute('select * from contact')
        print(self.c.fetchall())
    
    def commit(self):
        self.conn.commit()
        print("입력값 영구히 저장")
       
    def rollback(self):
        self.conn.rollback()
        print("입력값 영구히 취소")
       
    def close(self):
        self.c.close()
        self.conn.close()
        print("contact db close")

def set_contact11():
    name = input("이름을 입력하세요 : ")
    pn = input("핸드폰번호를 입력하세요 : ")
    email = input("메일을 입력하세요 : ")
    addr = input("주소를 입력하세요 : ")
    conIns = Contact11(name, pn, email, addr)
    conIns.print_info()
    conIns.open()
    conIns.table_check()
    conIns.input()
    conIns.commit()
    conIns.close()

set_contact11()

class Contact12:
    def __init__(self,name, pn, email, addr):
        self.name = name
        self.pn = pn
        self.email = email
        self.addr = addr

    def print_info(self):
        print("이름 : {} ".format(self.name))
        print("핸드폰번호 : {} ".format(self.pn))
        print("메일 : {} ".format(self.email))
        print("주소 : {} ".format(self.addr))
    
    def open(self):
        import sqlite3
        self.conn = sqlite3.connect('c:/data/contact.db')
        self.c = self.conn.cursor()
        print("contact db open")
    
    def table_check(self):
        #sqlite에서의 기능, table이 있으면 생성하지 않는다 
        self.c.execute('create table if not exists contact\
                       (name text,pn text, mail text,addr text)')               
        self.c.execute("PRAGMA table_info('contact')") 
        print(self.c.fetchall())
        
    def input(self):
        self.c.execute("insert into contact(name, pn, mail, addr)\
                       values(?,?,?,?)",(self.name,self.pn,self.email,self.addr))
        self.c.execute('select * from contact')
        print(self.c.fetchall())
    
    def commit(self):
        self.conn.commit()
        print("입력값 영구히 저장")
       
    def rollback(self):
        self.conn.rollback()
        print("입력값 영구히 취소")
       
    def close(self):
        self.c.close()
        self.conn.close()
        print("contact db close")

def set_contact12():
    name = input("이름을 입력하세요 : ")
    pn = input("핸드폰번호를 입력하세요 : ")
    email = input("메일을 입력하세요 : ")
    addr = input("주소를 입력하세요 : ")
    conIns = Contact12(name, pn, email, addr)
    conIns.print_info()
    conIns.open()
    conIns.table_check()
    conIns.input()
    conIns.commit()
    conIns.close()

set_contact12()

#[문제173] 한주간동안 걸음수를 요일별로 그래프를 그리세요.
#단 막대그래프 함수를 생성해서 인수값으로 걸음수, 요일을 입력하면 
#그래프가 그려지도록하세요.
#
#step = [5000,6000,7500,10000,10000,20000,2000]
#labels = ['월','화','수','목','금','토','일']
#cbc = create_bar_chart(step,labels,1)
#cbc.create_bar_chart()

class create_bar_chart:
    
    def __init__(self, xlabel, ylabel):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot = 0
        
    def create_chart(self, plot):
        import matplotlib.pylab as plt
        from matplotlib import font_manager, rc
        font_name = font_manager.FontProperties \
        (fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
        rc('font',family = font_name)
        self.plot = plot
        
        if self.plot == 1:
            plt.pie(self.ylabel, labels = self.xlabel)
            plt.title("일별 걸음 수")
            plt.show()
        elif self.plot == 2:
            plt.bar(xlabel, ylabel)
            plt.title("일별 걸음 수")
            plt.show()
        elif self.plot == 3:
            plt.barh(xlabel, ylabel)
            plt.title("일별 걸음 수")
            plt.show()
        elif self.plot == 4:
            plt.plot(xlabel, ylabel)
            plt.title("일별 걸음 수")
            plt.show()
        else:
            pass
    

xlabel = ['월','화','수','목','금','토','일']
ylabel = [5000,6000,7500,10000,10000,20000,2000]    
cbc = create_bar_chart(xlabel, ylabel)  
cbc.create_chart(1)
cbc.create_chart(2)
cbc.create_chart(3)
cbc.create_chart(4)

##선생님의 풀이 ##
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(
        fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

class create_bar_chart:
    def __init__(self,data, labels, bar):
        self.data = data
        self.labels = labels
        self.bar = bar
        
    def create_bar_chart(self):
    
        if self.bar == 1:
            plt.bar(self.labels, self.data, align='center')
            plt.xlabel('요일')
            plt.ylabel('걸음수')
        elif self.bar == 2:
            plt.barh(self.labels, self.data, align='center')
            plt.xlabel('걸음수')
            plt.ylabel('요일')
        elif self.bar == 3:
            plt.plot(self.labels,self.data,linestyle=':',color='r')
            plt.xlabel('요일',fontsize=10)
            plt.ylabel('걸음수',fontsize=10)
        elif self.bar == 4:
            self.step_max=max(self.data)
            #가장 큰 데이터만 조금 중심에서 벗어나게 하자 
            self.explode = [0.1 if i == self.step_max else 0.0 for i in self.data]
            plt.figure(figsize=(6,6))
            plt.pie(self.data,labels=self.labels,explode=self.explode,
                    opct='%1.1f%%',textprops={'fontsize':10})
            
        plt.title('한주간 동안 걸음수') 
        plt.grid()
        plt.show()
    
if __name__=='__main__':
    step = [5000,6000,7500,10000,10000,20000,2000]
    labels = ['월','화','수','목','금','토','일']
    cbc = create_bar_chart(step,labels,1)
    cbc.create_bar_chart()

if __name__=='__main__':
    step = [5000,6000,7500,10000,10000,20000,2000]
    labels = ['월','화','수','목','금','토','일']
    cbc = create_bar_chart(step,labels,2)
    cbc.create_bar_chart()

if __name__=='__main__':
    step = [5000,6000,7500,10000,10000,20000,2000]
    labels = ['월','화','수','목','금','토','일']
    cbc = create_bar_chart(step,labels,3)
    cbc.create_bar_chart()
    
if __name__=='__main__':
    step = [5000,6000,7500,10000,10000,20000,2000]
    labels = ['월','화','수','목','금','토','일']
    cbc = create_bar_chart(step,labels,4)
    cbc.create_bar_chart()

##################################################
class Viva:
    cnt = 0
    def __init__(self, name):
        self.name = name
        print("{} 님이 게임방에 들어왔습니다.".format(self.name))
        Viva.cnt += 1
        
    def count_viva(self):
        print("현재 {}명이 남았습니다.".format(Viva.cnt))
     
    #인스턴스를 삭제할때 작동함 
    def __del__(self):
        print("{} 님이 게임방에서 나갔습니다.".format(self.name))
        Viva.cnt -= 1
        
man1 = Viva("홍길동")
man1.count_viva()    

man2 = Viva("박찬호")
man2.count_viva()

man1.count_viva()

del man1

man2.count_viva()
man1.count_viva()#error

del man2

class Viva2:
    cnt = 0
    def __init__(self, name):
        self.name = name
        print("{} 님이 게임방에 들어왔습니다.".format(self.name))
        Viva2.cnt += 1
       
    @classmethod #클래스 메소드 지정 
    def count_viva(cls):#이 경우에는 self 대신 cls 넣어야 함 
        print("현재 {}명이 남았습니다.".format(cls.cnt))
     
    #인스턴스를 삭제할때 작동함 
    def __del__(self):
        print("{} 님이 게임방에서 나갔습니다.".format(self.name))
        Viva2.cnt -= 1

man1 = Viva2("홍길동")
man1.count_viva()    

man2 = Viva2("박찬호")
man2.count_viva()

man1.count_viva()

del man1

man2.count_viva()
man1.count_viva()#error

del man2

######
class Student:
    
    def __init__(self, name, id, grade, detail):
        self._name = name
        self._id = id
        self._grade = grade
        self._detail = detail
    
    def __str__(self):
        return '{} {}'.format(self._name, self._id)
    
student1 = Student('홍길동',1,1,{'성별':'남','영어':100})
student2 = Student('도바킨',1,1,{'성별':'남','영어':80})

#인스턴스의 인수값 보기 
student1.__dict__
#{'_name': '홍길동', '_id': 1, '_grade': 1, '_detail': {'성별': '남', '영어': 100}}
#인스턴스를 출력해 보자, __str__ 함수의 return 값이 출력된다. 
print(student1)#홍길동 1

lst = []
lst.append(student1)
lst.append(student2)
#보면 객체값이 들어가 있다. 
lst
print(lst)
#출력값을 보면 append 할때 __str__ 함수의 return 값이 들어가 있다. 
for i in lst:
    print(i)

##
class Student2:
    
    def __init__(self, name, id, grade, detail):
        self._name = name
        self._id = id
        self._grade = grade
        self._detail = detail
    
    def __repr__(self):
        return '{} {}'.format(self._name, self._id)
    
    def show(self):
        print("학생 정보 이름 : {}, 등급 : {}, 자세하게 : {}".format(
                self._name, self._grade, self._detail))
    
student1 = Student2('홍길동',1,1,{'성별':'남','영어':100})
student2 = Student2('도바킨',1,1,{'성별':'남','영어':80})
student1.show()

student1.__dict__
print(student1)

lst = []
lst.append(student1)
lst.append(student2)
#객체값이 아니라 __repr__ 함수의 리턴값이 들어가 있다. 
lst

#[문제 174] Sets 클래스 안에 union, intersection, difference 메소드를 생성하시요
x = {1,2,3,4,5,2,3,4,5}
y = {2,4,6,8,10}

for i in x:
    print(i)

class Sets:
    
    def __init__(self,set1, set2):
        self.set1 = set1
        self.set2 = set2
        print(type(self.set1))
        print(type(self.set2))
    
    def union(self):
        if (type(self.set1) is set) & (type(self.set2) is set):
            self.union_set = set()#{} 하면 dictionary 임 ...
        
            for i in self.set1:
                self.union_set.add(i)
        
            for i in self.set2:
                if i in self.union_set:
                    pass
                else:
                    self.union_set.add(i)
        
            return self.union_set
        
        elif (type(self.set1) is list) & (type(self.set2) is list):
            self.union_list = []
        
            self.union_list.extend(self.set1)
        
            for i in self.set2:
                if i in self.union_list:
                    pass
                else:
                    self.union_list.append(i)
        
            return self.union_list
            
        else:
            print("두 데이터 타입이 다릅니다.")
        
    def intersection(self):
        if (type(self.set1) is set) & (type(self.set2) is set):
            
            self.intersection_set = set()
        
            for i in self.set1:
                if i in self.set2:
                    self.intersection_set.add(i)
                else:
                    pass
        
            return self.intersection_set
        
        elif (type(self.set1) is list) & (type(self.set2) is list):
        
            self.intersection_list = []
        
            for i in self.set1:
                if i in self.set2:
                    self.intersection_list.append(i)
                else:
                    pass
        
            return self.intersection_list
            
        else:
            print("두 데이터 타입이 다릅니다.")
                
    def difference(self):
        if (type(self.set1) is set) & (type(self.set2) is set):
            self.difference_set = set()
        
            for i in self.set1:
                if i not in self.set2:
                    self.difference_set.add(i)
                else:
                    pass
        
            return self.difference_set
        
        elif (type(self.set1) is list) & (type(self.set2) is list):
            self.difference_list = []
        
            for i in self.set1:
                if i not in self.set2:
                    self.difference_list.append(i)
                else:
                    pass
        
            return self.difference_list
            
        else:
            print("두 데이터 타입이 다릅니다.")
##class Sets 끝 

x = {1,2,3,4,5,2,3,4,5}
y = {2,4,6,8,10}  
type(x)  
type(y)
s = Sets(x,y)
s.union()   
s.intersection()     
s.difference()

x = [1,2,4,5,6,3]
y = [3,5,6,3,2,7]
type(x) == 'list'
type(x) is list
s = Sets(x,y)
s.union()   
s.intersection()     
s.difference()

## 선생님의 풀이 ##
class Sets():                  
    def __init__(self,a,b):     
        self.a = a             
        self.b = b            
       
    def union(self):
        self.result = []                   
        for i in self.a:  
            if i not in self.result:    
                self.result.append(i) 
        
        for i in self.b:  
            if i not in self.result:    
                self.result.append(i) 
        
        self.result.sort()
        return self.result
    
    def intersection(self):    
        self.result = []
        for i in self.a:
            if i in self.b:
                self.result.append(i)
        self.result.sort()
        return self.result
    
    def difference(self):
        self.result = []
        for i in self.a:    
            if i not in self.b:    
                self.result.append(i)   
        self.result.sort()
        return self.result
    
x = [1,5,6,3,7]
y = [2,4,5,3,7]

s = Sets(x,y)                
print(s.__dict__)
s.union()
s.intersection()
s.difference()

######################################################################
#10/11#
u = [1,2,3,4,5,6,7,8,9]#전체집합
a = [2,4,6,8]
#a의 여집합은?

a_c = []

for i in u:
    if i not in a:
        a_c.append(i)
    else:
        continue
a_c

##class의 인스턴스에 있는 변수들을 출력해보자 
class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
p1 = Point(1,3)
p2 = Point(2,4)

#객체값이 나온다. 
print(p1)
print(p2)

#class 에 함수를 추가하자 
class Point2:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def show(self):
        print('({}, {})'.format(self.x, self.y))

p1 = Point2(1,3)
p2 = Point2(2,4)
#객체값이 나온다.
print(p1)
print(p2)
#함수를 이용해서 값을 볼 수 있다. 
p1.show()

#class에 함수를 추가하자 
class Point3:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def show(self):
        print('({}, {})'.format(self.x, self.y))
    
    def __str__(self):
        return '({}, {})'.format(self.x, self.y)

p1 = Point3(1,3)
p2 = Point3(2,4)
#__str__ 함수 return 값이 나온다.
print(p1)
print(p2)

#두개 인스턴스에 들어있는 변수들을 더하려면 ?
class Point4:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    def add(self,z):
        new_x = self.x + z.x
        new_y = self.y + z.y
        return (new_x, new_y)
    
    def show(self):
        print('({}, {})'.format(self.x, self.y))
    
    def __str__(self):
        return '({}, {})'.format(self.x, self.y)

p1 = Point4(1,3)
p2 = Point4(2,4)
#__str__ 함수 return 값이 나온다.
print(p1)
print(p2)
#add()를 추가했음 
p1.add(p2)
#다음은 안되는데 
p1 + p2
#class에 새로운 함수들을 추가해보자 
class Point5:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    #+를 사용할 수 있다. 
    def __add__(self,z):
        new_x = self.x + z.x
        new_y = self.y + z.y
        return (new_x, new_y)
    
    def __sub__(self,z):
        new_x = self.x - z.x
        new_y = self.y - z.y
        return (new_x, new_y)

    def __mul__(self,z):
        new_x = self.x * z.x
        new_y = self.y * z.y
        return (new_x, new_y)
    
    def show(self):
        print('({}, {})'.format(self.x, self.y))
    
    def __str__(self):
        return '({}, {})'.format(self.x, self.y)

p1 = Point5(1,3)
p2 = Point5(2,4)
#__str__ 함수 return 값이 나온다.
print(p1)
print(p2)
#__add__()를 추가했음
p1 + p2
#__sub__()를 추가했음
p1 - p2
#__mul__()를 추가했
p1 * p2

#tuple
t1 = ('도바킨',10,'남')
type(t1)
print(t1)

t2 = ('트레이서',12,'여')
type(t2)
print(t2)

for i in [t1,t2]:
    print("%s은(는) %d 세의 %s성 입니다."%i)
    
#namedtuple() #원소에 지시자가 붙은 tuple 생성 
import collections
Person = collections.namedtuple("Person",'name age gender')
n1 = Person(name = "홍길동",age = 10, gender = "남")
n2 = Person(name = "김태희",age = 40, gender = "여")
print(n1)
print(n2)

for i in [n1,n2]:
    print('%s은(는) %d세의 %s성 입니다.'%i)

#직교좌표계의 두 점이라 하자  
t1 = (1.0,6.0)
t2 = (3.5,2.5)
#두 점 사이의 거리 
distance = ((t2[0] - t1[0])**2 + (t2[1] - t1[1])**2)**(1/2)

#index로 표시하지 말고 다르게 표시를 해 보자 
from collections import namedtuple
Point = namedtuple('Point','x y')
#값이 중복되거나, 예약어를 쓰면 안 된다. 예약어를 사용할 경우 rename = True 하기 
Point = namedtuple('Point','x y class',rename = True)
#Point = namedtuple('Point',['x', 'y'])
#Point = namedtuple('Point','x,y')

p1 = Point(1.0,6.0)
p2 = Point(3.5,2.5)

Point.__dict__
#그럼 다시 두 점 사이의 거리를 구하면 
distance = ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**(1/2)
distance

#dictionary를 보자 
dict1 = {}
dict1["numpy"] = "과학 계산용 라이브러리"
dict1["pandas"] = "데이터 처리와 분석 라이브러리"
dict1["matplotlib"] = "시각화 라이브러리 "
print(dict1)

for i, j in dict1.items():
    print(i,j)
    
dict2 = {}
dict2["matplotlib"] = "시각화 라이브러리 "
dict2["numpy"] = "과학 계산용 라이브러리"
dict2["pandas"] = "데이터 처리와 분석 라이브러리"

#입력순서가 다르지만 두 dictionary 는 같다. 
dict1 == dict2#True

#입력순서가 다르면 두 dictionary가 달라지게 해 보자 
import collections
dict3 = collections.OrderedDict()
dict3["numpy"] = "과학 계산용 라이브러리"
dict3["pandas"] = "데이터 처리와 분석 라이브러리"
dict3["matplotlib"] = "시각화 라이브러리 "

dict4 = collections.OrderedDict()
dict4["pandas"] = "데이터 처리와 분석 라이브러리"
dict4["matplotlib"] = "시각화 라이브러리 "
dict4["numpy"] = "과학 계산용 라이브러리"

print(dict3)
print(dict4)

dict3 == dict4#False !!

#변수에 값 지정하기 
t = (1,2)
x ,y = t

lst = ["라인하르트",60,'독일',(1990,10,10)]
name, age, addr, birthday = lst
print(name)
name, age, addr, (year, month, day) = lst

str = "happly"
a,b,c,d,e,f = str

name, _, addr, _ = lst
print(name)
print(_)#마지막것만 나옴 

#* : 언패킹하기, 원하는 정보만 골라서 추출할때 도움 됨 
name, age, *info = lst
print(name)
print(*info)

*info, pn = lst
print(*info)

#
str1 = "바스티온/23/4590093849803/010101-3748382/010-1111-1111/서울"
name, old, *_, addr = str1.split('/')
print(name)
print(old)
print(addr)

name, *_, (year, *_) = lst
print(name)
print(year)

#[문제 176] cal_stock 함수를 생성하세요
my_stock = [("삼성전자",100,49000),("현대차",50,126000),('LG전자',200,68800)]

def cal_stock(lst):
    
    all_price = 0
    
    for i in lst:
        all_price += i[1] * i[2]
    
    return all_price


cal_stock(my_stock)

#[문제 177] cal_stock 함수 만들기, namedtuple 사용
from collections import namedtuple

def cal_stock2(lst):
    
    all_price = 0
    Stock = namedtuple("stock","company num price")
    
    for i in lst:
        s = Stock(*i)
        all_price += s.num * s.price
    
    return all_price

cal_stock2(my_stock)

#tuple의 값을 바꾸기 
Stock = namedtuple('stock',['name','amount','price'])

s = Stock("NAVER",100,1500000)
s.amount
s._replace(amount = 150)
s

def show1(*arg):
    
    for i in arg:
        print(i)
        
show1('도바킨',100,1000)

#dictionary를 가변으로 넣을때 
def show2(**dic):
    print(dic)
    print(dic.keys())
    print(dic.values())
    
    for key, value in dic.items():
        print("{}, {}".format(key, value))
        
show2(name = "한조", 점수 = 10000, 급여 = 2300000)

#arg1 : [], () arg2 : {}
def show3(*arg1, **arg2):
    print(arg1)
    print(arg2)
    
show3(1,2)
#(1, 2) 
#*arg1로만 데이터가 들어감
show3(name = "한조", 점수 = 10000, 급여 = 2300000)
#()
#{'name': '한조', '점수': 10000, '급여': 2300000}
#**arg2로만 데이터가 들어감 
show3(1,2,name = "한조", 점수 = 10000, 급여 = 2300000)
#(1, 2)
#{'name': '한조', '점수': 10000, '급여': 2300000}

x = {'name':'겐지','power':100,'인기':20000}
show3(x)
#({'name': '겐지', 'power': 100, '인기': 20000},)
#{}
show3(**x)
#()
#{'name': '겐지', 'power': 100, '인기': 20000}

#list의 원소의 갯수를 해야려 보자 
x = ['a','b','c','a','c','d','b']
dict1 = {}

for i in x:
    if i in dict1.keys():
        dict1[i] += 1
    else:
        dict1[i] = 1
        
dict1

dict1 = {}
for i in x:
    if i not in dict1.keys():
        dict1[i] = 0
    dict1[i] += 1

dict1

dict1 = {}
for i in x:
    #setdefault(i,0) :i 있으면 넘어감, 없으면 등록 
    dict1.setdefault(i,0)#dict1에 i가 있는지 봄, 없으면 i 등록하고 value는 0
    dict1[i] += 1

dict1


import collections
cnt = collections.Counter(x)
print(cnt)

dict2 = collections.defaultdict(int)#value의 type
dict2

#원소의 갯수가 세어지는 과정을 볼 수 있다. 
for i in x:
    dict2[i] += 1
    print(dict2)
    
#dictionary의 key에 대해 여러 value 들을 넣어보자  
dict31 = {'info':'02'}
dict32 = {'info':'서울'}

dict1 = {}
dict1.setdefault('info',[]).append('02')
dict1.setdefault('info',[]).append('서울')
dict1

#value가 list이다
dict4 = collections.defaultdict(list)
dict4['info'].append('02')
dict4['info'].append('서울')
dict4

#value가 set이다 
dict4 = collections.defaultdict(set)
dict4['info'].add('02')
dict4['info'].add('서울')
dict4

#value가 tuple인건 안 된다
dict4 = collections.defaultdict(tuple)
dict4['info'].append('02')
dict4['info'].append('서울')
dict4

##반복문을 사용해서 해 보자 
dictionary = {}
x1 = {'info':'02'}
x2 = {'info':'서울'}
x3 = {'info':'010-1000-0001'}
x4 = {'info2':'051'}
x5 = {'info2':'부산'}
x6 = {'info2':'010-1001-0002'}

#values가 list일때 
for i in [x1,x2,x3,x4,x5,x6]:
    for k, v in i.items():
        if k not in dictionary:
            dictionary[k] = [v]
        else:
            dictionary[k].append(v)
  
print(dictionary)

#
dictionary = collections.defaultdict(list)
for i in [x1,x2,x3,x4,x5,x6]:
    for k, v in i.items():
        if k not in dictionary:
            dictionary[k] = [v]
        else:
            dictionary[k].append(v)

print(dictionary)

#values 가 set일때
dictionary = collections.defaultdict(set)
for i in [x1,x2,x3,x4,x5,x6]:
    for k, v in i.items():
        if k not in dictionary:
            dictionary[k] = set()
            dictionary[k].add(v)
        else:
            dictionary[k].add(v)

print(dictionary)

