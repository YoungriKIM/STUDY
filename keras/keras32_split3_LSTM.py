# 과제 및 실습 LSTM으로 모델 만들기
# 지금까지 배운 전처리 등 다 적용
# 데이터 1~100 / x의 열은 5개, 6은 1 /
# predict 만들 것 (96,97,98,99,100)->101 ~ (100,101,102,103,104)->105 >> 예상 predict는 (101,102,103,104,105)

#32-4 Dense과 비교

import numpy as np

#1. 데이터

a = np.array(range(1,101))
size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):     #행
        subset = seq[i : (i+size)]           #열
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset.shape) #(95, 6)

x = dataset[:, :5]
y = dataset[:, 5]
print(x.shape) #(95,6)
print(y.shape) #(95,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape) #(60, 5, 1)
print(x_val.shape) #(16, 5, 1)
print(x_test.shape) #(19, 5, 1)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(20, input_shape=(5,1), activation='relu'))
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=20, mode='min')

model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_data=(x_val, y_val), verbose=2, callbacks=[stop])

#평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=5)
print('loss: ', loss)

# predict 만들 것 (96,97,98,99,100)->101 ~ (100,101,102,103,104)->105 >> 예상 predict는 (101,102,103,104,105)

b = np.array(range(96,105))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

x_pred = split_x(b, size)
# print(x_pred)
# print(x_pred.shape) #(5,5)

x_pred = scaler.transform(x_pred)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)
y_pred = model.predict(x_pred)
print('y_pred: ', y_pred)

# 32-2 LSTM 모델
# 268/1000
# loss:  [0.0041620442643761635, 0.05484079197049141]
# y_pred:  [[101.39189 ]
#  [102.484604]
#  [103.584076]
#  [104.69028 ]
#  [105.80326 ]]