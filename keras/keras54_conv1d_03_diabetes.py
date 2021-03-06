# Conv1D로 바꾸시오 그리고 비교하시오

import numpy as np
from sklearn.datasets import load_diabetes #당뇨병 수준

#데이터 불러오고 , 이건 사이킷런의 데이터 불러오는 방법
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) #(442, 10) (442,)

#트레인 테스트 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state=66)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle=True, random_state=66)

#데이터 전처리 (MinMaxScaler를 이용해서 , 기준은 x_train으로)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(200, 1, input_shape=(10,1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(100, 1))
model.add(Conv1D(80, 1))
model.add(Flatten())
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

#컴파일, 훈련 (Earlystopping 적용)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping])

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss, mae: ', loss, mae)

y_predict = model.predict(x_test)

#RMSE와 R2 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
      return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)



#1파일
# loss, mae:  3136.7783203125 44.4537353515625
# RMSE:  56.00695024986382
# R2:  0.409904405219381

#2파일
# loss, mae:  2492.786865234375 40.156646728515625
# RMSE:  49.92781650416666
# R2:  0.5682723462876598

#3파일
# loss, mae:  2428.074951171875 40.22421646118164
# RMSE:  49.275496967583756
# R2:  0.5575202675507116

#4파일
# loss, mae:  3150.124267578125 45.089866638183594
# RMSE:  56.12596891470034
# R2:  0.5600951159333354

#5파일
# loss, mae:  3332.93212890625 45.96597671508789
# RMSE:  57.731553418894926
# R2:  0.4032728564644378

#6파일
# loss, mae:  3121.9990234375 46.306800842285156
# RMSE:  55.87484721026777
# R2:  0.5189554519135346

# 54-3 Conv1D
# loss, mae:  4252.28662109375 54.935550689697266
# RMSE:  65.20955071350713
# R2:  0.34479829972209697