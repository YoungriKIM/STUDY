# 21-1의 predict 값이 소수점이 아니라 0.1 로 나오게 할 것

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
# print(x.shape) #(569, 30)
# print(y.shape) #(569,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

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
model.add(Dense(120, activation='relu', input_shape=(30,)))
model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(2, activation='sigmoid'))

#4. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=15, mode='max')

model.fit(x_train, y_train, epochs=2000, batch_size=15, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print('loss, acc: ', loss, acc)

y_predict = model.predict(x_test[-5:-1])
print('y_test[-5:-1]: ', y_test[-5:-1])
print('y_predict: ', y_predict)


# loss, acc:  0.11559246480464935 0.9824561476707458
# y_test[-5:-1]:  [0 1 0 1]
# y_predict:  [[1.23250155e-14] > 0.5미만 > 0
#  [9.83964145e-01] > 0.5이상 > 1
#  [5.24317222e-12] > 0.5미만 > 0
#  [9.95472968e-01]] > 0.5이상 > 1
#> 소수점이 나오는 이유는 sigmoid가 0~1사이이기 때문이다. 0.5이상은 1, 0.5미만은 0으로 바꾸는 작업이 필요하다.