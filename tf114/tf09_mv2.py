# 이번에는 데이터를 행렬단위로 넣어보자

import tensorflow as tf
tf.set_random_seed(66)

x_data = [[73, 80, 75],[93, 88, 93], 
          [85, 91, 90],[96, 98, 100],[73, 66, 70]]
y_data = [[152],[185],[180],[196],[142]]


# 행렬 데이터를 넣을 때는 쉐이프를 제대로 적어라
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = x * w + b 
# 으로 돌아가지 않는다.
hypothesis = tf.matmul(x, w) + b
# matmul 이 행렬 전용 곱셈 기능이다. / matrix multiplication

# [실습] 나머지 만들어 봐라 (verbose = step, cost, hypothesis)

cost = tf.reduce_mean(tf.square(hypothesis-y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
        if step % 10 == 0 :
            print(step, "cost(loss) : ", cost_val, "\n예측값: ", hy_val)



''' 성훈오빠꺼
cost = tf.reduce_mean(tf.square(hypothesis - y)) # loss='mse'

train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost) # optimizer + train
# train = tf.train.GradientDescentOptimizer(learning_rate=0.17413885).minimize(cost) # optimizer + train

# with문 사용해서 자동으로 sess가 닫히도록 할수도 있다.
with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(5001):
        cost_val, w_val, b_val, hy_val, _ =sess.run([cost,w,b,hypothesis,train], feed_dict={x:x_data,y:y_data})
        if step %20 == 0:
            print(step, cost_val, w_val, b_val) # epoch, loss, weight, bias
            print(step, "cost :", cost_val, "\n", hy_val)
'''