# 옵티마이저 부분 한 줄로 바꾸고
# with문 활용하여 세션 클로즈 자동으로

import tensorflow as tf

tf.set_random_seed(66)  # 랜덤값을 고정시켜놓았다

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]), name='weight') # random_normal : normal 정규분포화 된 random 값을 [1] 1개 넣어라
b = tf.Variable(tf.random_normal([1]), name='bias')

# W, b 잘 들어갔는지 출력해서 확인(하려면 세션 필요해서 불러왔다)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(W), sess.run(b))

# 1차 함수 식을 만들어보자
hypothesis = x_train * W + b

# loss 를 계산할 mse 식으로 만들어보자 / loss = mse
loss = tf.reduce_mean(tf.square(hypothesis - y_train))

# 이 손실을 계속 최적으로 만들어야 하니 경사하강법을 써야한다. 즉 optimizer를 할 것이다!
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)

# 위의 두줄을 한 줄로도 줄일 수 있다.
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 세스런해서 fit해보자 + 배리에이블 형태의 자료형식은 만들고 나서 꼭 초기화 해줘야 해!
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())


# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:      # 2000번 중에서 20번 마다 출력해주겠다는 의미
#         print(step, sess.run(train), sess.run(W), sess.run(b))

# --------------------------------------------------------------------------------------------
# 세션을 열었을 때는 세션을 다시 닫아줘야 한다.
# sess.close()
# 이렇게 까지 닫아줘야 알아서 메모리가 닫힌다. 자동으로 되기는 하지만 수동으로 하는 습관을 가지자
# 그런데 수동으로 하기 번거로우니까 with문을 써보자. 알아서 닫아준다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:      # 2000번 중에서 20번 마다 출력해주겠다는 의미
            print(step, sess.run(train), sess.run(W), sess.run(b))
