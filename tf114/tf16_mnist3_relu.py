# 이전 파일 직접 실습한 것과 선생님이 만들어 준 것과 비교!
# nan 값 나오는 것 수정!

# 텐서플로 1.14 부터 이미 케라스가 들어가 있어 keras 불러와서 쓸 수 있다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# 데이터 지정 + 스케일링
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float')/255.
x_test = x_test.reshape(10000, 28*28).astype('float')/255.

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (60000, 784) (60000, 10)
# (10000, 784) (10000, 10)

# 모델 구성
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
w1 = tf.Variable(tf.random_normal([784, 100], stddev=0.1), name='weight1')  # stddev=0.1 랜덤한 숫자의 분포를 10%로. 디폴트는 1
b1 = tf.Variable(tf.random_normal([1, 100], stddev=0.1), name='bias1')
# layer1 = tf.nn.softmax(tf.matmul(x, w1) + b1) # 소프트맥스 말고 렐루로 넣어주자
# layer1 = tf.nn.selu(tf.matmul(x, w1) + b1)    # 셀루도 가능
# layer1 = tf.nn.elu(tf.matmul(x, w1) + b1)       # 엘루도 가능
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# 드랍 아웃을 넣어보자!(0.3 만큼 버리기)
layer1 = tf.nn.dropout(layer1, keep_prob=0.3)

# 다층으로 구성하자!
w2 = tf.Variable(tf.random_normal([100, 50], stddev=0.1), name='weight2')
b2 = tf.Variable(tf.random_normal([1, 50], stddev=0.1), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([50, 10], stddev=0.1), name='weight3')
b3 = tf.Variable(tf.random_normal([1, 10], stddev=0.1), name='bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer2, w3) + b3)

# 컴파일 훈련(다중분류)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(201):
        _, cur_loss = sess.run([train, loss], feed_dict = {x:x_train, y:y_train})
        if epoch%10 == 0:
            y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
            y_pred = np.argmax(y_pred, axis = 1)
            print(f'Epoch {epoch} \t===========>\t loss : {cur_loss}')

    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis = 1)

    print('accuracy score : ', accuracy_score(y_test, y_pred))

# =======================================
# 계속해서 nan 값 나와서 tf.random_normal([50, 10], stddev=0.1) 로 랜덤구간 지정해줌
# accuracy score :  0.8826