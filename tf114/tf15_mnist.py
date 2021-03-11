# [실습] 엠니스트 만들어라
# 안되면 케라스 2.3.1

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (60000, 784) (60000, 1)
# (10000, 784) (10000, 1)

# vectorize
hot = OneHotEncoder()
hot.fit(y_train)
y_train = hot.transform(y_train).toarray()
y_test = hot.transform(y_test).toarray()
print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) 
# (60000, 784) (10000, 784)
# (60000, 10) (10000, 10)

# 데이터 형식 지정 + input_layer
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# 첫번째 레이어
w1 = tf.Variable(tf.zeros([784, 64]), name = 'weight1')
b1 = tf.Variable(tf.zeros([1, 64]), name = 'bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

# 두번째 레이어
w2 = tf.Variable(tf.zeros([64, 16]), name = 'weight2')
b2 = tf.Variable(tf.zeros([1, 16]), name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

# 아웃풋 레이어
w3 = tf.Variable(tf.zeros([16,1]), name='weight3') 
b3 = tf.Variable(tf.zeros([1,10]), name='bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer2, w3) + b3)

# 컴파일의 loss = categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# 옵티마이저 지정
train = tf.train.AdamOptimizer(learning_rate=0.8).minimize(loss)

# 0~1 사이로 나올 프레딕트를 만들자
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(501):
        _, cost_val = sess.run([train, loss], feed_dict={x:x_train, y:y_train})
        if step % 20 == 0:
            print(step, cost_val)

    # 프레딕트 해보자
    y_pred = sess.run(predicted, feed_dict={x:x_test})
    # print(y_pred)
    # print(y_test)
    print("accuracy_score: ", sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# ===============================
# accuracy_score:  0.9


''' 포동재꺼
# [실습]
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
tf.set_random_seed(66)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000,)

x_train = x_train.reshape(x_train.shape[0], 784)/255.
x_test = x_test.reshape(x_test.shape[0], 784)/255.

y_train = to_categorical(y_train)

x = tf.placeholder(tf.float32, shape = (None, 784))
y = tf.placeholder(tf.float32, shape = (None, 10))

#2. 모델
w1 = tf.Variable(tf.random.normal([784, 256], stddev= 0.1, name = 'weight1'))
b1 = tf.Variable(tf.random.normal([1, 256], stddev= 0.1, name = 'bias1'))
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random.normal([256, 128], stddev= 0.1, name = 'weight2'))
b2 = tf.Variable(tf.random.normal([1, 128], stddev= 0.1, name = 'bias2'))
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random.normal([128, 64], stddev= 0.1, name = 'weight3'))
b3 = tf.Variable(tf.random.normal([1, 64], stddev= 0.1, name = 'bias3'))
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)

w4 = tf.Variable(tf.random.normal([64, 10], stddev= 0.1, name = 'weight4'))
b4 = tf.Variable(tf.random.normal([1, 10], stddev= 0.1, name = 'bias4'))
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

#3. 컴파일, 훈련, 평가
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(251):
        _, cur_loss = sess.run([train, loss], feed_dict = {x:x_train, y:y_train})
        if epoch%10 == 0:
            y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
            y_pred = np.argmax(y_pred, axis = 1)
            print(f'Epoch {epoch}\t===========>\t loss : {cur_loss} \tacc : {accuracy_score(y_test, y_pred)}')

    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis = 1)

    print('accuracy score : ', accuracy_score(y_test, y_pred))

# accuracy score :  0.9738

# accuracy score :  0.9765  >> adam 0.01 / 251 / epoch
'''