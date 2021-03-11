# tf06_2.py의 lr을 수정해서
# epoch가 2000보다 작게 만들어라

import tensorflow as tf

tf.set_random_seed(66)  # 랜덤값을 고정시켜놓았다

# x,y 트레인 데이터를 placeholder로 --------------------------------------------
# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None]) # shape=[None]도 돌아감
y_train = tf.placeholder(tf.float32, shape=[None])
# --------------------------------------------------------------------------------

W = tf.Variable(tf.random_normal([1]), name='weight') # random_normal : normal 정규분포화 된 random 값을 [1] 1개 넣어라
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate=0.17413875).minimize(loss)


# 샘이 수정해준 것 --------------------------------------------------------------------------------------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
    if step % 20 == 0:
        print(step, loss_val, W_val, b_val)
        if [2.0000000] >= W_val >= [1.9990000] : break

# 프레딕트를 넣어라  --------------------------------------------------------------------------------------------
aa = [4]
bb = [5, 6]
cc = [6, 7, 8]

# 위의 학습에서 W, b가 갱신되었기 때문에 x_train만 바꿔주면 된다. x_train은 이미 플레이스 홀더니 값을 feed로 넣어준다.
print(sess.run(hypothesis, feed_dict={x_train:aa}))
print(sess.run(hypothesis, feed_dict={x_train:bb}))
print(sess.run(hypothesis, feed_dict={x_train:cc}))

# [8.998177]
# [10.997122 12.996066]
# [12.996066 14.99501  16.993954]

# [실습] lr 줄여서 최적의 에폭 줄이기 ---------------------------------------------------------------------------------
# learning_rate=0.1
# 200 7.46e-07 [1.9990209] [1.0022255]

