import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate

#데이터를 주자
x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])
y1 = np.array([range(711,811), range(1,101), range(201,301)])

#행렬을 바꿔줘야 함
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

#트레인이랑 테스트 분리
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size=0.8, shuffle=True)

#모델을 구상하자 인풋에 2개 병합 1개 아웃풋 1개겠네
#모델1

input1 = Input(shape=(3,))
dense1 = Dense(4)(input1)
dense2 = Dense(16)(dense1)
dense2 = Dense(32)(dense1)

input2 = Input(shape=(3,))
dense2 = Dense(7)(input2)
dense2 = Dense(14)(dense2)
dense2 = Dense(48)(dense2)

merge1 = concatenate([dense1, dense2])
middle1 = Dense(2)(merge1)
middle1 = Dense(6)(middle1)
middle1 = Dense(8)(middle1)
middle1 = Dense(14)(middle1)
middle1 = Dense(20)(middle1)
output1 = Dense(3)(middle1)

model = Model(inputs = [input1, input2], outputs = output1)
model.summary()

#컴파일,훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1_train, epochs=10, batch_size=1, validation_split=0.2, verbose=0)

#평가, 예측
