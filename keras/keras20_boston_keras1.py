# 2개의 파일을 만드시오
# 1. Earlystopping 을 적용하지 않는 최고의 모델
# 2. Earlystopping 을 적용한 최고의 모델

#아까는 사이킷런 이번에는 케라스에서 제공, 케라스의 데이터 셋을 불러오는 방법 찾아서 적용
import numpy as np

#1 데이터 주고
from tensorflow.keras.datasets import boston_housing

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

x_train = train_data
y_train = train_target
x_test = test_data
y_test = test_target

print(x_train.shape) #(404, 13)
print(y_train.shape) #(404,)
print(x_test.shape) #(102, 13)
print(y_test.shape) #(102,)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense1 = Dense(300, activation='relu')(input1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(100)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=300, batch_size=8, validation_data=(x_val, y_val), verbose=1)

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=2)
print('loss, mae: ', loss, mae)

y_predict = model.predict(x_test)

#RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)

#1번파일
# loss, mae:  11.377790451049805 2.260841131210327
# RMSE:  3.3730985125849697
# R2:  0.8633197073667653