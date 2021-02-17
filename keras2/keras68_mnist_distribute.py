# gpu가 2개일때 (2way) 분산처리 하도록 코드를 추가하자!
# https://keras.io/guides/distributed_training/


import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.    
x_test = x_test.reshape(10000, 28, 28, 1)/255. 
 
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

# y 리쉐잎
y_train = y_train.reshape(y_train.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# y 벡터화 OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
hot = OneHotEncoder()
hot.fit(y_train)
y_train = hot.transform(y_train).toarray()
y_test = hot.transform(y_test).toarray()
y_val = hot.transform(y_val).toarray()

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# gpu 두개 사용하기 코드 추가! -------------------------------------------------------
# 개수에 상관 없이 모두 돌아가게 된다. 선택적으로 하고 싶으면 공식문서 확인
import tensorflow as tf
strategy = tf.distribute.MirroredStrategy(cross_device_ops=\
    tf.distribute.HierarchicalCopyAllReduce())
# ----------------------------------------------------------------------------------

with strategy.scope():
    model = Sequential()
    model.add(Conv2D(filters=120, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(100, 2, strides=1))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(80, 2, strides=1))
    model.add(Flatten())
    model.add(Dense(60))
    model.add(Dense(60))
    model.add(Dense(60))
    model.add(Dense(30))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=30, batch_size=28, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=28)
print('loss, acc: ', loss, acc)

y_pred = model.predict(x_test[:10])

# print('y_pred: ', y_pred.argmax(axis=1))
# print('y_test: ', y_test[:10].argmax(axis=1))


#==================
# 기록용
# 40-2 mnist CNN
# loss, acc:  0.0900002047419548 0.90000319480896        21
# loss, acc:  0.010415063239634037 0.9835000038146973     17
# loss, acc:  0.009324220940470695 0.9854999780654907     69
# # y_pred:  [7 2 1 0 4 1 4 9 5 9]
# y_test:  [7 2 1 0 4 1 4 9 5 9]
