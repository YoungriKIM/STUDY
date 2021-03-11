# 배치를 정해주자!
# 튜닝해서 애큐러시 0.9이상으로

# 튜닝용 파일

import tensorflow as tf 

# 텐서플로 1.14 부터 이미 케라스가 들어가 있어 keras 불러와서 쓸 수 있다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


# 데이터 지정 + 스케일링
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float')/255.
x_test = x_test.reshape(10000, 28*28).astype('float')/255.

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (60000, 784) (60000, 10)
# (10000, 784) (10000, 10)

# 모델 구성
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

# ------------------------------------------------------------------------------------------------------------
# Variable, get_variable 은 비슷하지만 initializer를 추가할 수 있다.
# w1 = tf.Variable(tf.random_normal([784, 100], stddev=0.1), name='weight1')
w1 = tf.get_variable('weight1', shape=[784, 100], initializer=tf.contrib.layers.xavier_initializer())
# 레이어를 출력해보자
print('w1: ', w1)
# w1:  <tf.Variable 'weight1:0' shape=(784, 100) dtype=float32_ref>

# ------------------------------------------------------------------------------------------------------------
b1 = tf.Variable(tf.random_normal([1, 100], stddev=0.1), name='bias1')
print('b1: ', b1)
# b1:  <tf.Variable 'bias1:0' shape=(1, 100) dtype=float32_ref>

# ------------------------------------------------------------------------------------------------------------
# layer1 = tf.nn.softmax(tf.matmul(x, w1) + b1) # 소프트맥스 말고 렐루로 넣어주자
# layer1 = tf.nn.selu(tf.matmul(x, w1) + b1)    # 셀루도 가능
# layer1 = tf.nn.elu(tf.matmul(x, w1) + b1)     # 엘루도 가능

layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)       
print('layer1: ', layer1)
# layer1:  Tensor("Elu:0", shape=(?, 100), dtype=float32)

# 드랍 아웃을 넣어보자!(0.3 만큼 버리기)
# layer1 = tf.nn.dropout(layer1, keep_prob=0.3)
# print('layer1: ', layer1)
# layer1:  Tensor("dropout/mul_1:0", shape=(?, 100), dtype=float32)

# ------------------------------------------------------------------------------------------------------------
w2 = tf.get_variable('weight2', shape=[100, 128], initializer=tf.compat.v1.initializers.he_normal())
b2 = tf.Variable(tf.random_normal([1, 128], stddev=0.1), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
# layer2 = tf.nn.dropout(layer2, keep_prob=0.3)

# ------------------------------------------------------------------------------------------------------------
w3 = tf.get_variable('weight3', shape=[128, 64], initializer=tf.compat.v1.initializers.he_normal())
b3 = tf.Variable(tf.random_normal([1, 64], stddev=0.1), name='bias3')
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
# layer3 = tf.nn.dropout(layer3, keep_prob=0.3)

# ------------------------------------------------------------------------------------------------------------
w4 = tf.get_variable('weight4', shape=[64, 10], initializer=tf.compat.v1.initializers.he_normal())
b4 = tf.Variable(tf.random_normal([1, 10], stddev=0.1), name='bias4')
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

# ------------------------------------------------------------------------------------------------------------
# 컴파일 훈련(다중분류)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

# ------------------------------------------------------------------------------------------------------------
# 배치사이즈를 안 정하면 6만개씩 한 번에 들어간다. 배치사이즈를 정해주자

training_epochs = 200                        # 에폭 지정
batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 60000/100 = 600 > 1에폭당 600번*100개씩 > 1에폭당 6만개는 동일한데 배치사이즈에 따라서 1에폭당 몇개씩 몇번 돌릴지가 달라진다.
print(total_batch) # 200
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):    # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, train], feed_dict=feed_dict)
        avg_cost += c/total_batch   # += : 오른쪽의 값을 매번 갱신해서 더해줌(600번 더해질 것임) / avg_cost: average cost 모두 더한 c를 600으로나누니 평균값이 나온다.

    print('Epoch: ', '%04d' % (epoch+1),
          'cost = {:.9f}'.format(avg_cost))

print('===== done =====')

# ------------------------------------------------------------------------------------------------------------
prediction = tf.equal(tf.math.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc: ', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))


# ==============================
# Acc:  0.9751