# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:11:50 2019

@author: STU24
"""

#12/2#
import numpy as np
#conda install -c menpo opencv
import cv2 as cv
print(cv.__version__)
#3.4.1
#conda install -c conda-forge opencv ... 안되는뎅 

#image file을 읽어들이자
img = cv.imread("C:\\WorkSpace\\dog test 1.jpg",0)
img

#아래 3개 같이 실행 
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()


#선분을 그려보자
#color 설정
blue_color = (255,0,0)
green_color = (0,255,0)
red_color = (0,0,255)
white_color = (255,255,255)

#배경이 검정색인 빈 Canvas 생성
img = np.zeros((384,384,3), np.uint8)

#파란색 선분을 그려보자
#line(배경, 시작점, 종료점, 색상, 선 두께) 
img = cv.line(img, (10,10),(20,20), blue_color,5)

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()

#배경이 흰색인 빈 Canvas 생성
img = np.zeros((384,384,3), np.uint8) + 255

#녹색 선분을 그려보자 
img = cv.line(img, (20,30), (50,60), green_color,4)

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()

#화살표 선분을 그려보자

img = np.zeros((384,384,3),np.uint8) + 255

#arrowedLine(배경, 시작점, 종료점, 색상, 화살표크기)
img = cv.arrowedLine(img, (50,50),(250,250),green_color, thickness = 2)

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()

#사각형을 그려보자 

img = np.zeros((384,384,3),np.uint8) + 255

#rectangle(배경,시작점, 종료점, 색상, 선 두께)
img = cv.rectangle(img, (10,10),(90,90),green_color,4)
img = cv.rectangle(img, (100,100),(190,190),blue_color,5)
img = cv.rectangle(img, (210,210),(290,290),red_color,6)

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()

#두께 = -1이면 내부가 색칠된 사각형을 얻을 수 있다. 
img = np.zeros((384,384,3),np.uint8) + 255

img = cv.rectangle(img, (10,10),(90,90),green_color,-1)
img = cv.rectangle(img, (100,100),(190,190),blue_color,-1)
img = cv.rectangle(img, (210,210),(290,290),red_color,-1)

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()

