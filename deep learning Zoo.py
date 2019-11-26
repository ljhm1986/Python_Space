# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:48:45 2019

@author: stu12
"""

#11/26#
import numpy as np
import pandas as pd
import tensorflow as tf

# http://archive.ics.uci.edu/ml/datasets/zoo
#   1. animal name:     (deleted)
#   2. hair     Boolean"
#   3. feathers     Boolean"
#   4. eggs     Boolean"
#   5. milk     Boolean"
#   6. airborne     Boolean"
#   7. aquatic      Boolean"
#   8. predator     Boolean"
#   9. toothed      Boolean"
#  10. backbone     Boolean"
#  11. breathes     Boolean"
#  12. venomous     Boolean"
#  13. fins     Boolean"
#  14. legs     Numeric (set of values: {0",2,4,5,6,8}) 
#  15. tail     Boolean"
#  16. domestic     Boolean"
#  17. catsize      Boolean"
#  18. type     Numeric (integer values in range [0",6])
# 1 : 포유류, 2 : 조류, 3 : 파충류, 4 : 어류, 5 : 양서류,
# 6 : 곤충/거미류, 7 : 무척추동물
# 곤충/거미류 들은 무척추동물에 속하지 않는가? 

#데이터를 보면 동물을 분류하는데 불필요한 것이 무엇인가? 필요한 독립변수만 
#포함하고, 불필요한 독립변수는 무엇인지 판단해서 제외하고 할 수 있어야 한다. 
#무엇이 필요하고, 무엇이 불 필요한지 구분하는데 도메인 지식이 필요하다.
 
zoo_data = np.loadtxt("C:/WorkSpace/Python_Space/data/zoo_data.txt",
                      delimiter = ',', usecols = range(1,18),
                      dtype = np.float32)

zoo_data


x_data = zoo_data[:,:-1]
x_data.shape
y_data = zoo_data[:,[-1]]
y_data.shape
type(y_data)#numpy.ndarray
x_data.dtype
y_data.dtype
y_data = np.int32(y_data)
#import collections
#collections.Counter(y_data)

x = tf.placeholder(tf.float32, shape = [None, 16])
y = tf.placeholder(tf.int32,#one hot encoding 하려면 int로 해야함에 주의 
                   shape = [None, 1])

y_one_hot = tf.one_hot(y,7)
y_one_hot = tf.reshape(y_one_hot, [-1,7])

w1 = tf.Variable(tf.random_normal([16,10]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([10]), name = 'bias1')
layer1 = tf.matmul(x, w1) + b1

w2 = tf.Variable(tf.random_normal([10,7]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([7]), name = 'bias2')

logits = tf.matmul(layer1,w2) + b2
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(
        logits = logits, labels = y_one_hot)
cost = tf.reduce_mean(cost_i)

#학습률 조정하기 
train = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)

predict = tf.argmax(hypothesis, 1)
correct_predict = tf.equal(predict, tf.argmax(y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(train, feed_dict = {x:x_data, y:y_data})
    
    if step % 500 == 0:
        print("cost_val")
        cost_v, acc = sess.run([cost, accuracy],
                               feed_dict = {x:x_data, y:y_data})
        print("step {:5}, cost : {:.3f}, acc : {:.1%}".format(step, cost_v, acc))
   
#loss, acc, hyp, logi = sess.run([cost, accuracy, hypothesis, logits],
#                                feed_dict ={x:x_data, y:y_data})
#print(loss, acc, hyp, logi)

#clam,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,7
a = sess.run(hypothesis, feed_dict = {x:
    [[0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0]]})
print(sess.run(tf.argmax(a,1))) #4

score = sess.run(hypothesis, feed_dict = {x:x_data})
print(score)
print(sess.run(tf.argmax(score,1)))
(sess.run(tf.argmax(score,1)).reshape(101,1) == y_data).sum() / 1.01

