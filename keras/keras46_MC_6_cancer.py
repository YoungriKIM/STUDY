# 21-1의 predict 값이 소수점이 아니라 0.1 로 나오게 할 것

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터 주고
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

# print(x.shape, y.shape) #(569, 30) (569,)
# print('x[:5]: ',x[:5]) #전처리가 안 되어있음
# print('y: ',y) #값이 0과 1로 이진분류임

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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
stop = EarlyStopping(monitor='loss', patience=10, mode='min')

modelpath = '../data/modelcheckpoint/k46_6_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
stop = EarlyStopping(monitor='val_loss', patience=5, mode='min') #monitor=val_loss도 가능, 그냥 loss보다 val_loss를 신뢰하기도 한다.
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_data=(x_val, y_val), verbose=2, callbacks=[stop, mc])

#검증, 예측
loss = model.evaluate(x_test, y_test, batch_size=10)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])
print('y_predict: ', y_predict)


#시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(18,6))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.show()



# print('y_predict_argmax: ', y_predict.argmax(axis=1)) #0이 열, 1이 행
# print('y_test[-5:-1]: ',y_test[-5:-1])




