# 저장한 npy 파일을 불러와보자

import numpy as np

x = np.load('../data/npy/cancer_x.npy')
y = np.load('../data/npy/cancer_y.npy')


#전처리(y벡터화, 트레인테스트나누기, 민맥스스케일러)
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

#모델 구성
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(30,))
dense1 = Dense(120, activation='relu')(input1)
dense1 = Dense(120)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(60)(dense1)
output1 = Dense(2, activation='sigmoid')(dense1)
model = Model(inputs = input1, outputs = output1)

#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=5, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_data=(x_val, y_val), verbose=2, callbacks=[stop])

#검증, 예측
loss = model.evaluate(x_test, y_test, batch_size=10)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])

# print('y_predict: ', y_predict)
# print('y_predict_argmax: ', y_predict.argmax(axis=1)) #0이 열, 1이 행
# print('y_test[-5:-1]: ',y_test[-5:-1])

# 50-3 제대로 돌아가는 것 확인
# loss:  [0.11050102114677429, 0.9649122953414917]