# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:56:26 2019

@author: stu12
"""

###########################################################################
#9/24#
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
#2차원 array 합치기 
np.concatenate([x,y], axis = 0)#row합치기, column갯수 같아야 함
#array([[ 5,  7, 22],
#       [ 6, 54,  2],
#       [ 8, 77, 12],
#       [ 4, 66,  3]])
np.concatenate([x,y], axis = 1)#column합치기, row갯수 같아야 함 
#array([[ 5,  7, 22,  8, 77, 12],
#       [ 6, 54,  2,  4, 66,  3]])
np.vstack((x,y))
#array([[ 5,  7, 22],
#       [ 6, 54,  2],
#       [ 8, 77, 12],
#       [ 4, 66,  3]])
#np.vstack((x,y), axis = 1)

#######################################################################
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

z = np.array(range(1,37)).reshape((3,4,3))
z
z.size
z.ndim
z.itemsize
z.nbytes

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

############################################################################
#11/19#

#numpy 
#다차원 배열 객체를 제공
#과학적 계산, 데이터 분석을 이용할 때 사용하는 라이브러리
import numpy as np

a = np.array([1,2,3])
a
#array([1, 2, 3])
type(a)#numpy.ndarray
a.shape
#(3,)
a.ndim
#1
a.dtype
#dtype('int32')

a[0]
a[0] = 4 #수정하기
a
#array([4, 2, 3])

b = np.array([[1,2,3],[4,5,6]])
b
#array([[1, 2, 3],
#       [4, 5, 6]])
print(b)
#[[1 2 3]
# [4 5 6]]
b.shape
b.ndim
b.dtype
b[0]
b[0,0]
b[0][0]
b[:,0]

#0으로만 채워져 있는 
np.zeros((2,2))
np.zeros((4,5))
#1로만 채워져 있는
np.ones((2,2))

#해당 숫자로만 채워져 있는 
np.full((2,2),7)
np.full((4,3),5)

#대각 행렬
np.eye(2)
np.eye(4)
#임의의 값으로 채우기
np.random.random((2,2))
np.random.random((3,3))
np.empty((2,2))
np.empty((3,4))
np.empty((5,6))

#list -> array 로 바꾸기
data1 = [1,2,3,4,5]
type(data1)
data2 = np.array(data1)
type(data2)
data2.shape
data2.ndim
data2.dtype

data3 = [[1,2,3],[4,5,6]]
type(data3)
data4 = np.array(data3)
type(data4)

np.zeros((3,4))
np.zeros((2,3,4)) #면, 행, 열 : 3차원

np.zeros((4,3,2))
np.full((4,6,5),2)

np.random.random((4,3,2))

#인덱싱 
x = np.array(([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))
x
x.shape
x[0]
x[0,0]
x[:,2]
x[2,:]
x[0,0:2]
x[0:2,1]
x[0:2,2]

##요소 추출하기 
x = np.array([[1,2],[3,4],[5,6]])

#1,4,5 를 추출해보자
x[0,0],x[1,1],x[2,0]
#한 번에 추출하기 
x[[0,1,2],[0,1,0]] #[[행],[열]]
np.array([x[0,0],x[1,1],x[2,0]])

x[[0,0],[1,1]] #array
[x[0,1],x[0,1]] #list

x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
#1 6 7 11 추출하기
x[[0,1,2,3],[0,2,0,1]]
np.array([x[0,0],x[1,2],x[2,0],x[3,1]])

b = [0,2,0,1]
x[np.arange(4),b]

#
x[[1,2],[0,1]]

#모든 요소에 10을 더하기
x[np.arange(4),b] += 10
x[np.arange(4),b]
x
#array([[11,  2,  3],
#       [ 4,  5, 16],
#       [17,  8,  9],
#       [10, 21, 12]])

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

#요소별 합
x + y
np.add(x,y)
#요소별 차
x - y
np.subtract(x,y)
#요소별 곱
x * y
np.multiply(x,y)
#요소별 나누기
x/y
np.divide(x,y)

#요소별 제곱근
np.sqrt(x)
np.sqrt(y)

v = np.array([9,10])
w = np.array([11,12])
#행렬의 곱
v.dot(w)
np.dot(v,w)

np.dot(x,v)
np.dot(y,w)

#모든 요소의 합
np.sum(x)
np.sum(y)
np.sum(v)
np.sum(w)

print(x)
#[[1 2]
# [3 4]]
#행을 열 기준으로 합치기
np.sum(x, axis = 0)
#array([4, 6])
#열을 행 기준으로 합치기
np.sum(x, axis = 1)
#array([3, 7])

#각 열의 평균 구하기
np.mean(x, axis = 0)
#array([2., 3.])
#각 행의 평균 구하기 
np.mean(x, axis = 1)
#array([1.5, 3.5])

#분산
np.var(x)
np.var(y)

#표준편차
np.std(x)
np.std(y)

#누적합
np.cumsum(x)
#행 방향 누적합
np.cumsum(x, axis = 0)
#열 방향 누적합
np.cumsum(x, axis = 1)

#요소 곱
np.prod(x)
np.prod(x, axis = 0)
np.prod(x, axis = 1)

np.prod(x, keepdims = True)#결과가 숫자가 아니라 1 * 1 array로 나온다.
#array([[24]]) 

np.prod(x)

x = np.array([[1,2],[3,np.nan]])

np.sum(x)
#nan
np.nansum(x)
#6.0

np.prod(x)
#nan
np.nanprod(x)
#6.0

x
np.max(x)
np.max(x, axis = 0)
#array([ 3., nan])
np.nanmax(x, axis = 1)
#array([2., 3.])

#누적곱
np.cumprod(x)
#array([ 1.,  2.,  6., nan])
np.nancumprod(x)
#array([1., 2., 6., 6.])
np.nancumprod(x, axis = 0)
#array([[1., 2.],
#       [3., 2.]])
np.nancumprod(x, axis = 1)
#array([[1., 2.],
#       [3., 3.]])

#브로드캐스팅
#shape이 다른 배열간에도 산술연산이 가능하게 하는 메커니즘 
x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
y = np.array([1,0,1])

x + y
for i in x:
    print(i + y)
    
z = np.zeros((4,3))
z

#모든 요소가 0이고 x와 shape가 같은 z만들기
z = np.zeros_like(x)
z = np.zeros(x.shape)

x.shape[0]

#브로드캐스팅 처럼 하려면 
for i in range(x.shape[0]):
    z[i,:] = x[i,:] + y
    
z

#y를 4번 쌓기
z = np.tile(y,(4,1))
print(y,z)

#
x = np.array([2,3,4])
y = np.array([1,5,2])
#요소별 최대값
np.maximum(x,y)
#요소별 최소값
np.minimum(x,y)

#최대값이 있는 인덱스 번호
np.argmax(x)
#최소값이 있는 인덱스 번호
np.argmin(x)

x[np.argmax(x)]
x[np.argmin(x)]

x = np.array([[1,8,2],[3,2,5],[9,5,3]])
print(x)
#[[1 8 2]
# [3 2 5]
# [9 5 3]]

np.argmax(x)
np.argmin(x)

#열별 최대값이 있는 인덱스 번호
np.argmax(x, axis = 0)
#array([2, 0, 1], dtype=int64)

#행별 최대값이 있는 인덱스 번호
np.argmax(x, axis = 1)
#array([1, 2, 0], dtype=int64)

#열별 최소값이 있는 인덱스 번호
np.argmin(x, axis = 0)
#array([0, 1, 0], dtype=int64)

#행별 최소값이 있는 인덱스 번호
np.argmin(x, axis = 1)
#array([0, 1, 2], dtype=int64)

name = np.array(['a','b','a','b','c'])
np.unique(name)

ids = np.array([1,2,4,2,6,8,9,4,2,2,54,1,5,7,7,2,2,4,8,90,4,4,2])
np.unique(ids)
set(ids)
sorted(set(ids))

ids == 1
np.in1d(ids,[1])

x = np.array(['a','b','a','b','c','c'])
y = np.array(['a','b','a','b','d','b'])

#교집합
np.intersect1d(x,y)

#합집합
np.union1d(x,y)

#차집합
np.setdiff1d(x,y)

x = np.array([1,2,4,10,15,20])

#배열 원소간 차 구하기 
np.diff(x)
#array([1, 2, 6, 5, 5])
np.diff(x, n = 2)
#array([ 1,  4, -1,  0])
np.diff(x, n = 3)
#array([ 3, -5,  1])

y = np.array([[1,2,3,4],[5,6,7,8]])
np.diff(y)
#array([[1, 1, 1],
#       [1, 1, 1]])
np.diff(y, axis = 0)
#array([[4, 4, 4, 4]])
np.diff(y, axis =1)
#array([[1, 1, 1],
#       [1, 1, 1]])
np.diff(x,y,n=2)# 안 되넹
np.diff([x,y], n = 2) #안 되넹


#np 배열의 저장 
x = np.arange(10)
y = np.arange(100)

np.save("c:/WorkSpace/Python_Space/npfile/x.npy",x)
x2 = np.load("c:/WorkSpace/Python_Space/npfile/x.npy")
x2

np.savez("c:/WorkSpace/Python_Space/npfile/xy.npz",x = x,y = y)
z = np.load("c:/WorkSpace/Python_Space/npfile/xy.npz")
z#객체값 출력
z['x']
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
z['y']
#array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#       68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
#       85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

#저장할때 확장자의 이름을 잘 모르겠으면 .확장자 를 빼고 써도 된다.
np.save("c:/WorkSpace/Python_Space/npfile/x1",x)
np.savez("c:/WorkSpace/Python_Space/npfile/xy1",x = x,y = y)

#일반 연산의 속도와 numpy 연산 속도를 비교해보자
import time

size = 1000000

def original():
    start = time.time()
    x = range(size)
    y = range(size)
    z = []
    
    for i in range(len(x)):
        z.append(x[i] + y[i])
    
    end = time.time()
    
    return end - start

def numpy():
    start = time.time()
    x = np.arange(size)
    y = np.arange(size)
    z = x + y
    end = time.time()
    return end - start

t1 = original()
t2 = numpy()

print("original process time : %.5fs"%t1)
print("numpy process time : %.5fs"%t2)
#행렬을 사용해서 연산이 빠르다. 

#transpose
x = np.array([[1,2],[3,4]])
x.T
x

x = np.arange(10)
x
x.T
x.reshape((5,2))
x.reshape((5,2), order = 'C') #행 우선
x.reshape((5,2), order = 'F') #열 우선

y = np.arange(10).reshape((5,2))
y
#array([[0, 1],
#       [2, 3],
#       [4, 5],
#       [6, 7],
#       [8, 9]])


#펼치기
y.reshape(10,)
y.ravel()
y.ravel('C') #행 순서로 나열 
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y.ravel('F') #열 순서로 나열
#array([0, 2, 4, 6, 8, 1, 3, 5, 7, 9])
y.flatten()

#행렬 붙이기 
x = np.array([[1,2,3],[4,5,6]])
y = np.array([[7,8,9],[10,11,12]])

np.concatenate([x,y], axis = 0)# 행으로 붙이기
#array([[ 1,  2,  3],
#       [ 4,  5,  6],
#       [ 7,  8,  9],
#       [10, 11, 12]])
np.concatenate([x,y], axis = 1)# 열로 붙이기
#array([[ 1,  2,  3,  7,  8,  9],
#       [ 4,  5,  6, 10, 11, 12]])

np.vstack([x,y])
#array([[ 1,  2,  3],
#       [ 4,  5,  6],
#       [ 7,  8,  9],
#       [10, 11, 12]])
np.hstack([x,y])
#array([[ 1,  2,  3,  7,  8,  9],
#       [ 4,  5,  6, 10, 11, 12]])

#반복하기 
x = np.arange(3)
x
#array([0, 1, 2])
x.repeat(2)
#array([0, 0, 1, 1, 2, 2])
x.repeat([2,3,4])#요소 별로 반복 횟수를 다르게 지정한다.
#array([0, 0, 1, 1, 1, 2, 2, 2, 2])

#
z = np.random.randn(2,2)
z
z.repeat(2, axis = 0)
z.repeat(2, axis = 1)
z.repeat([2,3], axis = 1)
np.tile(z,2)

s = np.array([5,3,4,2,1])
s
s.argsort() #sort한 이후의 인덱스 번호 
#array([4, 3, 1, 2, 0], dtype=int64)
s[s.argsort()]
#array([1, 2, 3, 4, 5])

np.argsort(s) #sort한 이후의 인덱스 번호 , 오름차순
#array([4, 3, 1, 2, 0], dtype=int64)
np.argsort(-s) #sort한 이후의 내림차순 번호, 내림차순
#array([0, 2, 1, 3, 4], dtype=int64)

np.sort(s)#오름차순
#array([1, 2, 3, 4, 5])
np.sort(s)[::-1]#내림차순
#array([5, 4, 3, 2, 1])

s1 = np.array([[3,1,7],[2,8,4],[1,0,3]])
np.sort(s1)
np.sort(s1, axis = 1)#각 행 정렬
np.sort(s1, axis = 0)#각 열 정렬 

id(s1) #메모리 주소 확인
s2 = s1
id(s2) #같다!!!

x = np.array([0,1,2,3])
id(x)
y = x
id(y) #같다!!

x[0]
y[0] = 10
x[0]

#메모리를 다르게 복사해야 한다.
z = x.copy()
id(z)

x[0] = 100
x
y
z

#list로 바꾸기 
type(z)
z = z.tolist()
type(z)

x = np.array([1,4,6,8,10,13,16,20])
#15를 어디어 넣는게 좋을까?

x.searchsorted(15)
#6

y = np.array([1,4,6,16,10,13,8,20])
y.searchsorted(15)
#7
#정렬이 되어 있는 경우에 쓰는게 일반적이다.

np.insert(x,6,15)#x의 6번 index에 15를 넣기
x

a = np.arange(1,7).reshape(3,2)
a
np.insert(a,3,[7,8], axis = 0)
np.insert(a,1,[7],axis = 0)
#array([[1, 2],
#       [7, 7],
#       [3, 4],
#       [5, 6]])
np.insert(a,1,[9],axis = 1)
#array([[1, 9, 2],
#       [3, 9, 4],
#       [5, 9, 6]])


np.delete(a,1,axis = 0) #2행 삭제
np.delete(a,1,axis = 1) #2열 삭제

# 1 2 3      5 6
#(     ) * ( 7 8 ) = ?
# 4 5 6      9 10

#a * b x * y

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[5,6],[7,8],[9,10]])
a.dot(b)
np.dot(a,b)
a.shape
b.shape

z = np.array([5,6,7])
np.dot(a,z)