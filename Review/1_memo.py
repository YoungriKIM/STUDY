import numpy as np
import tensorflow as tf

tf.set_random_seed(66)

# 1. 데이터

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size) # 60000/100

x = tf.placeholder(tf.float32, [None, 28,28,1])
y = tf.placeholder(tf.float32, [None, 10])

# 2. 모델구성

# Layer1---------------------------------------------------------------------
w1 = tf.get_variable('w1', shape=[3,3,1,32])
# shape=[3,3,1,32] # 제일 앞에 있는게 커널사이즈 (3,3) , 1은 채널(흑백), 32가 filter = output 다음레이어로 전달되는 node의 갯수
L1 = tf.nn.conv2d(x,w1, strides=[1,1,1,1],padding='SAME') # (28,28,32) 으로 나간다(padding='same')
# Conv2D(32,(3,3), input_shape =(28,28,1))
print(L1)
# Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L1)
# Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# Conv2D(filter, kernel_size, input_shape)
# Conv2D(10,(2,2), input_shape=(7,7,1)) 파라미터의 갯수?
# number_parameters = filter * (kernel_h * kernel_w * channels  + 1) = 10 *(2*2 +1) = 50

# ksize=[1,2,2,1] = max_pooling(2,2) 옆에 1, 1은 그냥 쉐입맞춰주기 위한 것

# Layer2----------------------------------------------------------------------
w2 = tf.get_variable('w2', shape=[3,3,32,64])
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME') 
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)
# Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32) 

# Layer3-----------------------------------------------------------------------
w3 = tf.get_variable('w3', shape=[3,3,64,128])
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME') 
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)
# Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)

# Layer4------------------------------------------------------------------------
w4 = tf.get_variable('w4', shape=[3,3,128,64])
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME') 
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)
# Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

# Flatten----------------------------------------------------------------------
L_flat = tf.reshape(L4, [-1,2*2*64])
print('flatten : ', L_flat)
# flatten :  Tensor("Reshape:0", shape=(?, 256), dtype=float32)

# Layer5(Dense)-----------------------------------------------------------------
w5 = tf.get_variable('w5', shape=[2*2*64, 64], 
                     initializer= tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([64]), name = 'b5')
L5 = tf.nn.selu(tf.matmul(L_flat, w5) + b5) 
L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5)
# Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

# Layer6(Dense)-----------------------------------------------------------------
w6 = tf.get_variable('w6', shape=[64, 32], 
                     initializer= tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([32]), name = 'b6')
L6 = tf.nn.selu(tf.matmul(L5, w6) + b6) 
L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6)
# Tensor("dropout_1/mul_1:0", shape=(?, 32), dtype=float32)

# Layer7(Dense)-----------------------------------------------------------------
w7 = tf.get_variable('w7', shape=[32, 10], 
                     initializer= tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([10]), name = 'b7')
hypothesis = tf.nn.softmax(tf.matmul(L6, w7) + b7) 
print('최종 출력 : ', hypothesis)
# Tensor("Softmax:0", shape=(?, 10), dtype=float32)

# 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_mean(y*tf.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# 훈련

training_epochs = 100
batch_size = 100
total_batch = int(len(x_train)/batch_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss= 0

    for i in range(total_batch):  # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        l, _ = sess.run([loss,optimizer], feed_dict=feed_dict)
        avg_loss += l/total_batch
    print('epoch : ', '%04d' %(epoch+1),
          'loss = {:.9f}'.format(avg_loss))
print("끝")

prediction = tf.equal(tf.math.argmax(hypothesis, 1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))