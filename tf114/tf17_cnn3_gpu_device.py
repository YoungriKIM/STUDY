# 두 개 이상의 글카가 있을 떄 어떤 것을 쓸지 설정해보자
# keras68_mnist_distribute 는 한 파일에 여러개 글카를 쓰는 거!!

import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(66)

#--------------------------------------------------------------------------------------------------------
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())   # False
print(tf.__version__)           # 2.3.1
#--------------------------------------------------------------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try :
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e :
        print(e)

# gpus[1] 으로 해서 이 파일을 돌리고
# gpus[0] 으로 해서 keras68_mnist2_gpu_device 파일을 돌리면
# 한 번에 두 개 이상을 돌릴 수 있다.
# 단 글카도 두 개 이상 ^^(글카 종류가 달라도 문제 없음~)
#--------------------------------------------------------------------------------------------------------

#1.  데이터 + 전처리 + 형식 지정
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (60000, 28, 28, 1) (60000, 10)
# (10000, 28, 28, 1) (10000, 10)

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#2. 모델구성
#--------------------------------------------------------------------------------------------------------
# L1.
w1 = tf.compat.v1.get_variable('w1', shape=[3, 3, 1, 128])     # 3, 3 > kernel_size / 1 > chennel, input_dim / 32 > filters, output_dim
# == Conv2D(32, (3,3), input_shape=(28,28,1))

L1 = tf.compat.v1.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
# strides 도 동일한 4차원으로!하기 위해 [1,n,n,1] 으로 n,n이 실세고 양옆의 1은 맞추기용이다.
print('L1: ', L1) # L1:  Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
## 즉 연산이 되고 나면 (None,28,28,32)가 된다. padding=same이니까

L1 = tf.compat.v1.nn.relu(L1)

L1 = tf.compat.v1.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# ksize 도 동일한 4차원으로!하기 위해 [1,n,n,1] 으로 n,n이 실세고 양옆의 1은 맞추기용이다.
## 연산 되고 나면 (14,14,32)가 된다.
print('L1: ', L1)   # L1:  Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# bias는 지정 안 해요? : 디폴트로 들어가있단다~

#--------------------------------------------------------------------------------------------------------
# L2.
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 128, 64])     # 3, 3 > kernel_size / 1 > chennel, input_dim >> 이전 레이어의 아웃풋이 해당 레이어의 인풋이 되니까 32 / 32 > filters, output_dim
# == Conv2D(64, (3,3))
L2 = tf.compat.v1.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME') 
print('L2: ', L2)   # L2:  Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
L2 = tf.compat.v1.nn.relu(L2)
L2 = tf.compat.v1.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print('L2: ', L2)   # L2:  Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# L3.
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 64])     # 3, 3 > kernel_size / 1 > chennel, input_dim >> 이전 레이어의 아웃풋이 해당 레이어의 인풋이 되니까 64 / 32 > filters, output_dim
# == Conv2D(128, (3,3))
L3 = tf.compat.v1.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME')
print('L3: ', L3)   # L3:  Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
L3 = tf.compat.v1.nn.relu(L3)
L3 = tf.compat.v1.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print('L3: ', L3)   # L3:  Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# L4.
w4 = tf.compat.v1.get_variable('w4', shape=[3, 3, 64, 32])
L4 = tf.compat.v1.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME')
print('L4: ', L4)   # L4:  Tensor("Conv2D_3:0", shape=(?, 4, 4, 64), dtype=float32)
L4 = tf.compat.v1.nn.relu(L4)
L4 = tf.compat.v1.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print('L4: ', L4)   # L4:  Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# Flatten.
L_flat = tf.compat.v1.reshape(L4, [-1, 2*2*32])
print('L_flat: ', L_flat) # L_flat:  Tensor("Reshape:0", shape=(?, 256), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# L5.
w5 = tf.compat.v1.get_variable('w5', shape=[2*2*32, 16], initializer=tf.compat.v1.initializers.he_normal())
b5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16]), name='b5')
L5 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(L_flat, w5) + b5)
# L5 = tf.nn.dropout(L5, keep_prob=0.2)
print('L5: ', L5)   # L5:  Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# L6.
w6 = tf.compat.v1.get_variable('w6', shape=[16, 16], initializer=tf.compat.v1.initializers.he_normal())
b6 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16]), name='b6')
L6 = tf.compat.v1.nn.selu(tf.compat.v1.matmul(L5, w6) + b6)
# L6 = tf.nn.dropout(L6, keep_prob=0.2)
print('L6: ', L6)   # L6:  Tensor("dropout_1/mul_1:0", shape=(?, 32), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# L7.
w7 = tf.compat.v1.get_variable('w7', shape=[16, 10], initializer=tf.compat.v1.initializers.he_normal())
b7 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10]), name='b7')
hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(L6, w7) + b7)
print('hypothesis, L7: ', hypothesis)   # hypothesis, L7:  Tensor("Softmax:0", shape=(?, 10), dtype=float32)

#--------------------------------------------------------------------------------------------------------
# 3. 컴파일, 훈련

learning_rate = 1e-5
training_epochs = 100
batch_size = 64
total_batch = int(len(x_train)/batch_size)  # 60000 / 100

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
# Acc:  0.9831


# initializer=tf.compat.v1.initializers.variance_scaling()
# 이것도 써봐라~