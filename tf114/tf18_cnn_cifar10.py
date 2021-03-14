# 텐서플로 1.0 CNN으로 cifar10 을 만들어보자!

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
tf.compat.v1.set_random_seed(311)

#--------------------------------------------------------------------------------------------------------
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())   # False
print(tf.__version__)           # 2.3.1
#--------------------------------------------------------------------------------------------------------

#1.  데이터 + 전처리 + 형식 지정
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 10)
# (10000, 32, 32, 3) (10000, 10)

x = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#2. 모델구성
#--------------------------------------------------------------------------------------------------------
# L1.
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 3, 128])     # 3, 3 > kernel_size / 3 > chennel, input_dim / 128 > filters, output_dim
# == Conv2D(128, (3,3), input_shape=(32,32,3))
L1 = tf.compat.v1.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.compat.v1.nn.relu(L1)
L1 = tf.compat.v1.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print('L1: ', L1) 
# L1:  Tensor("MaxPool:0", shape=(None, 16, 16, 128), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# L2.
w2 = tf.compat.v1.get_variable('w2', shape=[2, 2, 128, 96])
L2 = tf.compat.v1.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME') 
L2 = tf.compat.v1.nn.relu(L2)
# L2 = tf.compat.v1.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print('L2: ', L2) 
# L2:  L2:  Tensor("Relu_1:0", shape=(None, 16, 16, 96), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# L3.
w3 = tf.compat.v1.get_variable('w3', shape=[2, 2, 96, 64]) 
L3 = tf.compat.v1.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.compat.v1.nn.relu(L3)
# L3 = tf.compat.v1.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print('L3: ', L3) 
# L3:  Tensor("Relu_2:0", shape=(None, 16, 16, 64), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# Flatten.
L_flat = tf.compat.v1.reshape(L3, [-1, 16*16*64])
print('L_flat: ', L_flat)
# L_flat:  Tensor("Reshape:0", shape=(None, 1024), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# L4.
w4 = tf.compat.v1.get_variable('w4', shape=[16*16*64, 64], initializer=tf.compat.v1.initializers.he_normal())
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64]), name='b4')
L4 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(L_flat, w4) + b4)
print('L4: ', L4)
# L4:  Tensor("Relu_3:0", shape=(None, 64), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# L5.
w5 = tf.compat.v1.get_variable('w5', shape=[64, 32], initializer=tf.compat.v1.initializers.he_normal())
b5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32]), name='b5')
L5 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(L4, w5) + b5)
print('L5: ', L5)
# L5:  Tensor("Selu:0", shape=(None, 32), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# L7.
w6 = tf.compat.v1.get_variable('w6', shape=[32, 10], initializer=tf.compat.v1.initializers.he_normal())
b6 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10]), name='b6')
hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(L5, w6) + b6)
print('hypothesis, L6: ', hypothesis)   # hypothesis, L7:  Tensor("Softmax:0", shape=(?, 10), dtype=float32)
# hypothesis, L6:  Tensor("Softmax:0", shape=(None, 10), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# 3. 컴파일, 훈련

learning_rate = 1e-4
training_epochs = 50
batch_size = 64
total_batch = int(len(x_train)/batch_size)  # 50000 / 100

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))    # categorical_crossentropy
train = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

# 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

print('==== traing start ====')

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, train], feed_dict=feed_dict)
        avg_cost += c/total_batch

    print('Epoch: ', '%04d' % (epoch+1),
          'cost = {:.9f}'.format(avg_cost))

print('===== traing done =====')

# 예측
prediction = tf.equal(tf.math.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc: ', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# ==============================================================
# Acc:  0.5602
# Acc:  0.6497