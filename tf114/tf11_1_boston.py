# 회귀모델데이터셋 불러와서 만들기
# [실습] 만들어라! 결과는 sklearn의 r2_score/accuracy_score를 사용할 것

from sklearn.datasets import load_boston
import tensorflow as tf
import numpy as np

# 데이터 지정 ----------------------------------------------------------------
tf.set_random_seed(66)

dataset = load_boston()
x_data = dataset.data
y = dataset['target']
y_data = y.reshape(-1,1)

print(x_data.shape)
print(y_data.shape)
# (506, 13)
# (506, 1)

# 데이터 형식 지정 ----------------------------------------------------------------
x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([13,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 레이어로 이해
hypothesis = tf.matmul(x, w) + b

# compile(loss='mse', optimizer='adam')
loss = tf.reduce_mean(tf.square(hypothesis-y))
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# 세션 생성
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 에폭 지정 해서 훈련
for step in range(2001):
    loss_val, hy_val,_ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if step % 50 == 0 :
        print(step, "cost(loss) : ", loss_val)#, "\n예측값: ", hy_val)