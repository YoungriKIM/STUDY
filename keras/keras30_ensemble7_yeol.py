#29-1 이용해서 행이 다른 앙상블 모델에 대해 공부하시오

import numpy as np
from numpy import array

#1. 데이터 x1 10개 / x2 13개로 앙상블 해보자
x1 = np.array([[1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9], [9,10], [10,11], [20,30], [30,40], [40,50]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70], [60,70,80], [70,80,90], [80,90,100], [90,100,110], [100,110,120], [2,3,4], [3,4,5], [4,5,6]])
y1 = array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70], [60,70,80], [70,80,90], [80,90,100], [90,100,110], [100,110,120], [2,3,4], [3,4,5], [4,5,6]])
y2 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65])
x2_predict = array([65,75,85])

print(x1.shape) #(13,2)
print(x2.shape) #(13,3)
print(y1.shape) #(13,3)
print(y2.shape) #(13,)
print(x1_predict.shape) #(2,)
print(x2_predict.shape) #(3,)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) #(13, 2, 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) #(13, 3, 1)
x1_predict = x1_predict.reshape(1, 2, 1) #(1, 2, 1)
x2_predict = x2_predict.reshape(1, 3, 1) #(1, 3, 1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, shuffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x1_train, x1_val, y1_train, y1_val = train_test_split(x1_train, y1_train, train_size=0.8, shuffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x2_train, x2_val, y2_train, y2_val = train_test_split(x2_train, y2_train, train_size=0.8, shuffle=True, random_state=311)


#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate

input1 = Input(shape=(2,1))
lstm1 = LSTM(27, activation='relu')(input1)
dense1 = Dense(18, activation='relu')(lstm1)
dense1 = Dense(9, activation='relu')(dense1)

input2 = Input(shape=(3,1))
lstm2 = LSTM(27, activation='relu')(input1)
dense2 = Dense(18, activation='relu')(lstm1)
dense2 = Dense(9, activation='relu')(dense1)

merge1 = concatenate([dense1, dense2])
middle = Dense(18)(merge1)
middle = Dense(18)(middle)

output1 = Dense(18)(middle)
output1 = Dense(9)(output1)
output1 = Dense(3)(output1)

output2 = Dense(18)(middle)
output2 = Dense(9)(output2)
output2 = Dense(1)(output2)

model = Model(inputs = [input1, input2], outputs = [output1, output2])

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=20, mode='min')

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=1000, batch_size=3, validation_data=([x1_val, x2_val], [y1_val, y2_val]), verbose=2, callbacks=[stop])

#평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=3)
print('loss: ', loss)

y1_pred, y2_pred= model.predict([x1_predict, x2_predict])
print('y1_pred: ', y1_pred)
print('y2_pred: ', y2_pred)


# 29-1 두 데이터의 행이 13으로 같음
# 436/1000
# loss:  0.05432714894413948
# y_pred:  [[85.81537]]

# 30-6 두 데이터의 행이 13과 10으로 다름 > 행을 맞추지 않으면 되지 않는다. > 자르거나 채울 수 있는 다른 방법을 찾아보자
# ValueError: Data cardinality is ambiguous:
#   x sizes: 6, 8
#   y sizes: 6, 8

#30-7 두 데이터의 열이 다름 > 열이 다른 건 된다.
# loss:  [5.882585048675537, 4.469101428985596, 1.4134836196899414]
# y1_pred:  [[7.091115  4.721337  4.5430965]]
# y2_pred:  [[94.68496]]

'''
앙상블 모델에 대한 궁금증
[Q1]
x1 = (13, 3, 1) 온도, 습도, 미세먼지 13일치
x2 = (13, 3, 1) 삼성, 카카오, 네이버 주가 13일치
[Q1-A] 가능한

[Q2]
x1 = (13, 2, 1) 온도, 습도 13일치
x2 = (13, 3, 1) 삼성, 카카오, 네이버 주가 13일치
[Q2-A] 가능한

[Q3]
x1 = (10, 3, 1) 온도, 습도, 미세먼지 10일치
x2 = (13, 3, 1) 삼성, 카카오, 네이버 주가 13일치
[Q3-A] 불가능한 > 행을 맞추려고 자르거나 프레딕트 한 값을 넣어주거나 할 수 있다. > 나중에 더 알려줌
'''
