# csv 파일 불러와서 만들어봐라!

import tensorflow as tf
import numpy as np
sess = tf.Session()

# 데이터 지정 ----------------------------------------------------------------
tf.set_random_seed(66)

dataset = np.loadtxt('../data/csv/data-01-test-score.csv', delimiter=',')

# print(sess.run(dataset))

x_data = dataset[:,:-1]
y_data = dataset[:,-1:]
# y_data = dataset[:, [-1]]

print(x_data.shape)
print(y_data.shape)

# 데이터 형식 지정 ----------------------------------------------------------------
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis-y))

train = tf.train.AdamOptimizer(learning_rate=0.8).minimize(cost)

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if step % 50 == 0 :
        print(step, "cost(loss) : ", cost_val)#, "\n예측값: ", hy_val)

# ================================================
# 2000 cost(loss) :  5.7379107

# ================================================
# [실습]
# 아래값 predict할 것
aa = [[73.,80.,75.]]
# 152.
bb = [[93.,88.,93.]]
# 185.
cc = [[89.,91.,90.]]
# 180.
dd = [[96.,98.,100.]]
# 196.
ee = [[73.,66.,70.]]
# 142.

print('aa pred: ',sess.run(hypothesis, feed_dict={x:aa}))
print('bb pred: ',sess.run(hypothesis, feed_dict={x:bb}))
print('cc pred: ',sess.run(hypothesis, feed_dict={x:cc}))
print('dd pred: ',sess.run(hypothesis, feed_dict={x:dd}))
print('ee pred: ',sess.run(hypothesis, feed_dict={x:ee}))

# ====================================
# aa pred:  [[152.61342]]
# bb pred:  [[185.07228]]
# cc pred:  [[181.77515]]
# dd pred:  [[199.73024]]
# ee pred:  [[139.18759]]