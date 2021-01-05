#LSTM을 이용해 시계열데이터의 결과값을 예측하자.
#tip. LSTM은 3차원의 x 데이터가 필요하다.
import numpy as np

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

# 전처리도 해줘야쥐(스케일링, 트레인테스트, 리쉐잎)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = x_pred.reshape(1, 3)
x_pred = scaler.transform(x_pred)

print(x.shape, y.shape, x_pred.shape) #(13, 3) (13,) (3,)
x = x.reshape(13, 3, 1)
x_pred = x_pred.reshape(1, 3, 1)
print(x.shape, y.shape, x_pred.shape) #(13, 3, 1) (13,) (1, 3, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(60, activation='relu', input_shape=(3,1)))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(54, activation='relu'))
model.add(Dense(54, activation='relu'))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=2000, batch_size=30, validation_data=(x_val,y_val), verbose=2, callbacks=[stop])

#평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)

y_pred = model.predict(x_pred)
print('y_pred: ', y_pred)

# 23-3
# loss:  0.82417231798172
# y_pred:  [[80.21291]]

# 27-1
# 138/1000
# loss:  0.07894691079854965
# y_pred:  [[80.02537]]

# 23-1 (review폴더)
# 1572/2000
# loss:  [0.3924308717250824, 0.5359042286872864]
# y_pred:  [[82.27247]]

