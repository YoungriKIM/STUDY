#이진분류에서 바뀌는 activation, loss에 집중

#실습1. acc 0.985이상
#실습2. 원래 y[-5:-1] 과 x[-5:-1]를 넣어서 나온 y_predicr값을 출력해서 비교해보자. #-1은 가장 끝의 값이란 뜻임 [:-1]은 0~전체임

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()

# print(datasets.DESCR) #이 두가지는 데이터셋을 받으면 꼭 확인해보자
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape) #(569, 30)
# print(y.shape) #(569,)

# print(x[:5])
# # [2.029e+01 1.434e+01 1.351e+02 1.297e+03 1.003e-01 1.328e-01 1.980e-01
# #   1.043e-01 1.809e-01 5.883e-02 7.572e-01 7.813e-01 5.438e+00 9.444e+01
# #   1.149e-02 2.461e-02 5.688e-02 1.885e-02 1.756e-02 5.115e-03 2.254e+01
# #   1.667e+01 1.522e+02 1.575e+03 1.374e-01 2.050e-01 4.000e-01 1.625e-01
# #   2.364e-01 7.678e-02]]
# print(y) #y의 값이 0아니면 1 , 지금까지는 실제 값이 나오는 회기모델 이번에는 분류모델
# # 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# #  1 1 1 1 1 1 1 0 0 0 0 0 0 1

#전처리 알아서
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=33)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=33)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(1, activation='relu', input_shape=(30,))) >> 이 한 줄로만 끝내면 히든이 없는 모델로 딥러닝이 아닌 래거시 모델이라고 한다.
model.add(Dense(120, activation='relu', input_shape=(30,)))
model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1, activation='sigmoid')) #이진분류에는 가장 마지막 activation를 꼭 sigmoid를 사용한다. 결과값이 무조건 0~1 사이에 수렴해야 하기 때문
#그럼 모델의 첫 레이어부터 다 써도 될까? 정답은 알아서 판단해라. 나중에는 자동화해서 할 수도 있다. 대신 시간이 더 걸린다.
#4. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) #acc = accuracy(정확도. 분류모델에 자주 쓰는 메트릭스) 1에 가까울 수록 좋다.
# 분류모델의 이진분류는 꼭 loss에 binary_crossentropy:0과1사이일 때 쓰는 지표로 낮을 수록 좋다.

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=15, mode='max')

model.fit(x_train, y_train, epochs=2000, batch_size=15, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print('loss, acc: ', loss, acc)

# y_predict = model.predict(x_test[-5:-1])
# print('y_test[-5:-1]: ', y_test[-5:-1])
# print('y_predict: ', y_predict)

# loss, acc:  0.11559246480464935 0.9824561476707458
# y_test[-5:-1]:  [0 1 0 1]
# y_predict:  [[1.23250155e-14] > 0.5미만 > 0
#  [9.83964145e-01] > 0.5이상 > 1
#  [5.24317222e-12] > 0.5미만 > 0
#  [9.95472968e-01]] > 0.5이상 > 1
#> 소수점이 나오는 이유는 sigmoid가 0~1사이이기 때문이다. 0.5이상은 1, 0.5미만은 0으로 바꾸는 작업이 필요하다.