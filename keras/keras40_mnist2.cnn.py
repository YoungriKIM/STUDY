# 인공지능계의 hello world라 불리는 mnist를 써보자

# 실습. 지표 acc로 완성하시오 0.99이상  / 컨블루션 2개 이상 / 
# 응용. y 테스트 10개와 프레딕트 10개를 출력하시오 

import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.           # 최대값인 255를 나누었으니 최소0~ 최대1이 된다. *float32이란 실수로 바꾼다는 뜻
x_test = x_test.reshape(10000, 28, 28, 1)/255.                             # 실수형이라는 것을 빼도 인식한다.
# x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))      # 실제로 코딩할 때는 이 방법이 가장 좋다!
 
# y 리쉐잎
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# y 벡터화 OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
hot = OneHotEncoder()
hot.fit(y_train)
y_train = hot.transform(y_train).toarray()
y_test = hot.transform(y_test).toarray()

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=300, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(100, 2, strides=1))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(100, 2, strides=1))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=16, mode='max')

model.fit(x_train, y_train, epochs=1000, batch_size=69, validation_split=0.2, verbose=2, callbacks=[stop])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=69)
print('loss, acc: ', loss, acc)

y_pred = model.predict(x_test[:10])

print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

# 기록용
# loss, acc:  0.0900002047419548 0.90000319480896        21
# loss, acc:  0.010415063239634037 0.9835000038146973     17

