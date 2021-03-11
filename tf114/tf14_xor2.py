# 텐서플로1에서 인공지능의 겨울 xor 문제를 알아보자
# 다층레이어를 만들어서 겨울을 지나보자~~

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(66)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 실습 만들어라!

# 인풋 레이어
x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

# 첫번째 레이어
# model.add(Dense(10, input_dim=2, activation=sigmoid))
w1 = tf.Variable(tf.random_normal([2,128]), name='weight1')  # [2,1] * [2,10] = [None,10]
b1 = tf.Variable(tf.random_normal([128]), name='bias1')      #  이니까 바이어스도 10개
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# 두번째 레이어(7개 해야지)
# model.add(Dense(7, activation=sigmoid))
w2 = tf.Variable(tf.random_normal([128,7]), name='weight2')
b2 = tf.Variable(tf.random_normal([7]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2) # 위에서 전달해준 layer1 이 x 를 대체한다.

# 아웃풋 레이어
# model.add(Dense(1, activation=sigmoid))
w3 = tf.Variable(tf.random_normal([7,1]), name='weight3')    # 마지막 레이어의 열은 y와 쉐잎이 같아야 하니
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)


# 그렇다면 loss 도 binary_crossentropy 로 바꿔야 한다.
# cost = tf.reduce_mean(tf.square(hypothesis - y))
# 앞에 마이너스 꼭 뭍여라!
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 0~1 사이로 나올 프레딕트를 만들자
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    # 예측은 모든 에폭가 돌고 해야해서 이렇게 빼준다.
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})
    print('예측값:\n', h, '\n원래값:\n', c, '\n정확도: ', a)

# ====================================
# 딥한 레이어 하기 전

# 예측값:
#  [[0.5098468 ]
#  [0.50287294]
#  [0.5002181 ]
#  [0.4932434 ]]
# 원래값:
#  [[1.]
#  [1.]
#  [1.]
#  [0.]]
# 정확도:  0.75

# ====================================
# 딥한 레이어로 바꿈

# 예측값:
#  [[0.24381457]
#  [0.7202913 ]
#  [0.90670615]
#  [0.10898118]]
# 원래값:
#  [[0.]
#  [1.]
#  [1.]
#  [0.]]
# 정확도:  1.0