# 실습: 다중분류 불러와서 만들어라
# accuracy_score로 결론 낼것

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
tf.set_random_seed(66)

# 데이터 불러오기
x_data, y_data = load_wine(return_X_y=True)
y_data = y_data.reshape(-1,1)
print(x_data.shape, y_data.shape)
# (178, 13) (178, 1)

# split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, shuffle = True, random_state = 311)

# vectorize
hot = OneHotEncoder()
hot.fit(y_train)
y_train = hot.transform(y_train).toarray()
y_test = hot.transform(y_test).toarray()
print(x_train.shape, x_test.shape)   # (142, 13) (36, 13)
print(y_train.shape, y_test.shape)   # (142, 3) (36, 1)

# 데이터 형식 지정 + input_layer
x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.zeros([13,3]), name = 'weight')
b = tf.Variable(tf.zeros([1,3]), name = 'bias')

# 소프트 맥스로 감싼 최종 레이어
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# 컴파일의 loss = categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# 옵티마이저를 만들어 loss를 최소화 시키자
train = tf.train.AdamOptimizer(learning_rate = 1e-5).minimize(loss)

# 0~1 사이로 나올 프레딕트를 만들자
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        _, cost_val = sess.run([train, loss], feed_dict={x:x_train, y:y_train})
        if step % 200 == 0:
            print(step, cost_val)

    # 프레딕트 해보자
    y_pred = sess.run(predicted, feed_dict={x:x_test})
    print(y_pred)
    print(y_test)
    print("accuracy_score: ", accuracy_score(y_test, y_pred))

# ======================================
# accuracy_score:  0.9166666666666666

# 동재가 만든 3개 이상에서 가장 큰값 1로 만들기
# np.where([a == np.max(a) for a in y], 1, 0)