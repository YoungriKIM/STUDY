# 엑티베이션 시그모이드도 하고 애큐러시지표도 만들자

import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1,2], [2,3], [3,1],
          [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0],
          [1], [1], [1]]    # 이진분류인 데이터

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 시그모이드를 추가하자
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
# 시그모이드로 결과 값을 덮어서 나오는 값을 한정시켜준다

# 그렇다면 loss 도 binary_crossentropy 로 바꿔야 한다.
# cost = tf.reduce_mean(tf.square(hypothesis - y))
# 앞에 마이너스 꼭 뭍여라!
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 0~1 사이로 나올 프레딕트를 만들자
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
# tf.cast: 조건에 따라 True이면 1, False이면 0을 출력한다. 이 경우 0.5 이상일 때 1을 출력한다.
# tf.equal:


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
# 4800 0.24966846
# 예측값:
#  [[0.07802196]
#  [0.195242  ]
#  [0.48302576]
#  [0.7082548 ]
#  [0.884784  ]
#  [0.9640333 ]]
# 원래값:
#  [[0.]
#  [0.]
#  [0.]
#  [1.]
#  [1.]
#  [1.]]
# 정확도:  1.0

