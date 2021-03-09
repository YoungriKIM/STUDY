# 선형 회귀 모델을 만들어보자
# y = wx + b
# >> x 는 placeholder 형태 / w, b는 variable로 해야겠지?

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
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# 이 손실을 계속 최적으로 만들어야 하니 경사하강법을 써야한다. 즉 optimizer를 할 것이다!
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# 손실이 적을 때가 최적이니가 옵티마이저에 cost를 넣고 최저일 때를 구하겠다.
train = optimizer.minimize(cost)

# 세스런해서 fit해보자 + 배리에이블 형태의 자료형식은 만들고 나서 꼭 초기화 해줘야 해!
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(2001):
    sess.run(train)
    if step % 20 == 0:      # 2000번 중에서 20번 마다 출력해주겠다는 의미
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# 참고로! 이 x,y 데이터에 맞는 W=2, b=1 이다.
# 출력물의 마지막 줄
# 2000 1.0781078e-05 [1.9961864] [1.0086691]
# 에폭    로스(mse)       W            b