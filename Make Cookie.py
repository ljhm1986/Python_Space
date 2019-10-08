# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:37:35 2019

@author: stu11
"""

class Cookie:
    #쿠키 공장 갯수 
    cookieFactoryCount = 0
    
    def __init__(self):
        #쿠키의 갯수 
        self.cookieCount = 0
        #저장된 쿠키의 갯수
        self.cookieSaveCount = 0
        Cookie.cookieFactoryCount += 1
        print("new Cookie Factory build")
        print("The number of Cookie Factory is {}"
              .format(Cookie.cookieFactoryCount))
        
    def cookieInfo(self):
        print("The number of Cookie Factory is {}".format(Cookie.cookieFactoryCount))
        print("The number of Cookie is {}".format(self.cookieCount))
        print("The number of saved Cookie is {}".format(self.cookieSaveCount))
    
    def makeCookie(self):
        self.cookieCount += 1
        print('make Cookie')
        print('The number of Cookie is {}'.format(self.cookieCount))
        
    def eatCookie(self):
        if self.cookieCount == 0:
            print("We have not Cookie")
        else:
            self.cookieCount -= 1
            print("Cookie is delicious !")
            print('The number of Cookie is {}'.format(self.cookieCount))

    def saveCookie(self):
        if self.cookieCount == 0:
            print("We have not Cookie")
        else:
            self.cookieSaveCount += 1
            self.cookieCount -= 1
            print("We save one Cookie")
    
    def loadCookie(self):
        if self.cookieSaveCount == 0:
            print("We have not saved Cookie")
        else:
            self.cookieSaveCount -= 1
            self.cookieCount += 1
            print("We load one Cookie")
    
            
cookie1 = Cookie()
cookie1
cookie1.makeCookie()
cookie1.eatCookie()
cookie1.saveCookie()
cookie1.cookieInfo()
