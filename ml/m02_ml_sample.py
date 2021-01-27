# 머신러닝이 딥러닝보다 먼저 생겼다. 머신러닝은 히든레이어가 없기때문에 gpu가 없는 환경에서도 돌아간다.
# keras 22_1_1 을 가져와서 씀 머신러닝 모델로 수정해보자.

import numpy as np
from sklearn.datasets import load_iris 

#1. 데이터 불러오기
dataset = load_iris()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# from sklearn.model_selection import train_test_split
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# from tensorflow.keras.utils import to_categorical     # 머신러닝은 y벡터화를 하지 않아도 된다.
# y = to_categorical(y)
# y_train = to_categorical(y_train)
# y_val = to_categorical(y_val)
# y_test = to_categorical(y_test)

# print(x.shape) #(150,4)
# print(y.shape) #(150,3)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import LinearSVC       # 머신러닝 모델을 먼저 불러온다.

# model = Sequential()
# model.add(Dense(120, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

model = LinearSVC()

#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# from tensorflow.keras.callbacks import EarlyStopping
# earlystopping = EarlyStopping(monitor='acc', patience=20, mode='max')
# model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=earlystopping)
model.fit(x_train,y_train)      # 머신러닝은 컴파일 없이 그냥 훈련이다.
# 발리데이션은 어떻게 넣지?

#4. 평가, 예측
# result = model.evaluate(x_test, y_test, batch_size=2)
# print('result: ', result)
result = model.score(x_test,y_test)
print('result: ', result)

# y_predict = model.predict(x_test[-5:-1])
# print('y_predict: ', y_predict)
y_predict = model.predict(x_test[-5:-1])
print('y_predict: ', y_predict)
print('y_test: ', y_test[-5:-1])

#'==========================='
# 딥러닝 모델
# loss:  [0.12436151504516602, 0.9666666388511658, 0.04672175273299217]

# 머신러닝 모델
# result:  0.9666666666666667     # model.score의 결과로 분류모델에서는 알아서 acc로 반환해준다.
