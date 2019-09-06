import math

def sumF(*x):
    total = 0
    for i in x:
        total = total + i
    return total

def mean(*x):
    sum_num = sumF(*x)
    result = sum_num/len(x)
    return result

def variance(*y):
    mid = mean(*y)
    dis = 0
    for i in y:
        dis = dis + (i - mid)**2
    result = dis/len(y)
    return result

def stddev(*z):
    var = variance(*z)
    std = math.sqrt(var)
    return std

