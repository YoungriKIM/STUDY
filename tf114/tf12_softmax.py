# 시그모이드를 했으니 이번에는 소프트맥스를 해보자!
# + loss = categorical_crossentropy

import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,6,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]   # 다중 분류의 데이터


x = tf.placeholder('float', [None,4])
y = tf.placeholder('float', [None,3])   # 시그모이드는 [none,1] 이었다.

# y와 덧셈을 하려면 w의 열이 3으로 같아야 한다.
w = tf.Variable(tf.random_normal([4, 3]), name='weight')
# 통상 바이어스는 한 레이어에 1개이지만 웨이트가 3열이기 때문에 1개씩 붙어서 1,3 이다.
# 헷갈릴 수도 있어서 추가! 단순 회귀와 이중 분류는 [1]인데, [1,1]로 넣어도 똑같다.
b = tf.Variable(tf.random_normal([1, 3]), name='bias')

# hypothesis = tf.matmul(x, w) + b
# (none,3) * (4,3) = (none,3) / (none,3) + (1 ,3) = (none,3) 즉 수식을 지나고 나면 y의 쉐이프와 같은 (none,3)이 된다.
# 이제 activation인 소프트맥스로 감싸주자!
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# 이렇게 최종 레이어까지 구성 완료

# loss = mse 이었는데 이번에는 다중분류를 위한 loss를 만들어보자
loss_mse = tf.reduce_mean(tf.square(hypothesis-y))
# loss = categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# 옵티마이저를 만들어 loss를 최소화 시키자
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # 프레딕트 해보자
    a = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
    print(a, sess.run(tf.arg_max(a, 1)))    # a 안에 있는 가장 높은 값에 1을 주어라
    # [[0.80384046 0.19088006 0.00527951]] [0] > 0번째가 가장 높다는 뜻

