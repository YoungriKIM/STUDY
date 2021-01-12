# 모델 세이브를 2번하여 비교해보는 파일이니 잘 읽어볼 것 

import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.           # 최대값인 255를 나누었으니 최소0~ 최대1이 된다. *float32이란 실수로 바꾼다는 뜻
x_test = x_test.reshape(10000, 28, 28, 1)/255.                             # 실수형이라는 것을 빼도 인식한다.
# x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))      # 실제로 코딩할 때는 이 방법이 가장 좋다!
 
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

# y 리쉐잎
y_train = y_train.reshape(y_train.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# y 벡터화 OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
hot = OneHotEncoder()
hot.fit(y_train)
y_train = hot.transform(y_train).toarray()
y_test = hot.transform(y_test).toarray()
y_val = hot.transform(y_val).toarray()

#2. 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model2 = Sequential()
model2.add(Conv2D(filters=120, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Dropout(0.2))
model2.add(Conv2D(100, 2, strides=1))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Dropout(0.2))
model2.add(Conv2D(80, 2, strides=1))
model2.add(Flatten())
model2.add(Dense(60))
model2.add(Dense(60))
model2.add(Dense(60))
model2.add(Dense(30))
model2.add(Dense(10, activation='softmax'))

# 모델 세이브를 하자. 모델이 끝난 지점에서 하면 모델만 저장된다.
# model.save('../data/h5/k52_1_model1.h5')


#3. 컴파일, 훈련
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

modelpath = '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'  #로 생성된 가장 마지막 파일 이름을 아래로 바꿔줌 
# k52_1_MCK_{val_loss:.4f}.hdf5
# d =  정수형으로 10의 자리까지 /f = float 실수형으로 소수 4번째까지 하겠다. ##이부분 찾아보기

stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

# hist = model.fit(x_train, y_train, epochs=30, batch_size=28, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])#, mc])

# 모델 세이브를 컴파일, 훈련 뒤에 하면 w값까지 저장된다.
# model.save('../data/h5/k52_1_model2.h5')
# model.save_weights('../data/h5/k52_1_weight.h5')

####### 2,3번 모두 없다고 생각하고 저장한 h5, hdf5을 불러와 보자 #######

model1 = load_model('../data/h5/k52_1_model2.h5')  # model1은 모델과 가중치를 불러왔다.

#4-1. 평가, 예측
result = model1.evaluate(x_test, y_test, batch_size=8)
print('model1_loss: ', result[0])
print('model1_acc: ', result[1])


model2.load_weights('../data/h5/k52_1_weight.h5')
#4-2. 평가, 예측
result = model2.evaluate(x_test, y_test, batch_size=8)
print('model2_loss: ', result[0])
print('model2_acc: ', result[1])

''' >> 모델세이브와 웨이트세이브에서 저장 된 w는 같아서 동일하다.
model1_loss:  0.05850084125995636
model1_acc:  0.982200026512146
model2_loss:  0.05850084125995636
model2_acc:  0.982200026512146
'''




#===================
# 기록용
# 40-2 mnist CNN
# loss, acc:  0.0900002047419548 0.90000319480896        21
# loss, acc:  0.010415063239634037 0.9835000038146973     17
# loss, acc:  0.009324220940470695 0.9854999780654907     69
# # y_pred:  [7 2 1 0 4 1 4 9 5 9]
# y_test:  [7 2 1 0 4 1 4 9 5 9]
