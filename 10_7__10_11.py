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