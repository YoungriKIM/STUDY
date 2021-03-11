# [살습]
# x,y 트레인 데이터를 placeholder로

import tensorflow as tf

tf.set_random_seed(66)  # 랜덤값을 고정시켜놓았다

# x,y 트레인 데이터를 placeholder로 --------------------------------------------
# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[3,]) # shape=[None]도 돌아감
y_train = tf.placeholder(tf.float32, shape=[3,])
# --------------------------------------------------------------------------------

W = tf.Variable(tf.random_normal([1]), name='weight') # random_normal : normal 정규분포화 된 random 값을 [1] 1개 넣어라
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 내가 수정 한 것 --------------------------------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        # sess.run(train)
        do = sess.run(train, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
        if step % 20 == 0:
            print(step, do, sess.run(W), sess.run(b))

# 샘이 수정해준 것 --------------------------------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        # sess.run(train)
        _, loss_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
        if step % 20 == 0:
            print(step, loss_val, W_val, b_val)


# sess 안의 것들의 순서가 바뀌어도 괜찮다.