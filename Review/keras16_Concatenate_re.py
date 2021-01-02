import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate
#1. 데이터 주기
x1 = np.array([range(100), range(100,200), range(200,300)])
y1 = np.array([range(1,101), range(101,201), range(201,301)])

x2 = np.array([range(5, 105), range(15,115), range(30,130)])
y2 = np.array([range(8,108), range(16, 116), range(32, 132)])
#(3,100)인 상태

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)
#(100,3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, x2_train, x2_test, y2_train, y2_test = train_test_split(x1, y1, x2, y2, train_size=0.8, shuffle=True)

#모델 구성
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(15)(dense1)
dense1 = Dense(20)(dense1)

input2 = Input(shape=(3,))
dense2 = Dense(4, activation='relu')(input2)
dense2 = Dense(8)(dense2)
dense2 = Dense(12)(dense2)

merge1 = Concatenate(axis=1)([dense1, dense2])
middle1 = Dense(6)(merge1)

output1 = Dense(3)(middle1)

output2 = Dense(3)(middle1)

model = Model(inputs=[input1, input2], outputs = [output1, output2])
model.summary()

#컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=10, batch_size=1, validation_split=0.2, verbose=1)

#평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print('loss: ', loss)
y1_predict, y2_predict2 = model.predict([x1_test, x2_test])
print('=================================')
print('y1_predict: ', y1_predict)


