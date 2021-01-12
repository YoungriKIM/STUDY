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

#=======================================================

#2. 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# model = Sequential()
# model.add(Conv2D(filters=120, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
# model.add(Conv2D(100, 2, strides=1))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
# model.add(Conv2D(80, 2, strides=1))
# model.add(Flatten())
# model.add(Dense(60))
# model.add(Dense(60))
# model.add(Dense(60))
# model.add(Dense(30))
# model.add(Dense(10, activation='softmax'))

# 모델 세이브를 하자. 모델이 끝난 지점에서 하면 모델만 저장된다.
# model.save('../data/h5/k52_1_model1.h5')


#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'  #로 생성된 가장 마지막 파일 이름을 아래로 바꿔줌 
# k52_1_MCK_{val_loss:.4f}.hdf5
# d =  정수형으로 10의 자리까지 /f = float 실수형으로 소수 4번째까지 하겠다. ##이부분 찾아보기

# stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
# mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

# hist = model.fit(x_train, y_train, epochs=30, batch_size=28, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])#, mc])

# 모델 세이브를 컴파일, 훈련 뒤에 하면 w값까지 저장된다.
# model.save('../data/h5/k52_1_model2.h5')
# model.save_weights('../data/h5/k52_1_weight.h5')

# model1 = load_model('../data/h5/k52_1_model2.h5')  # model1은 모델과 가중치를 불러왔다.

#4-1. 평가, 예측
# result = model1.evaluate(x_test, y_test, batch_size=8)
# print('model1_loss: ', result[0])
# print('model1_acc: ', result[1])

#---------------------------------------------------------

# model.load_weights('../data/h5/k52_1_weight.h5')
# 52-1에서 불러온 w로 평가 예측을 할 것이다. fit이 필요가 없다. 하지만 모델과 컴파일은 저장되지 않았다.

#4-2. 평가, 예측
# result = model.evaluate(x_test, y_test, batch_size=8)
# print('가중치_loss: ', result[0])
# print('가중치_acc: ', result[1])

# w가 준비되어 있어서 바로 결과가 나온다.
# 가중치_loss:  0.05850084125995636
# 가중치_acc:  0.982200026512146

#---------------------------------------------------------

# model2 = load_model('../data/h5/k52_1_model2.h5') # 이 파일은 모델과 w 까지 모두 저장되었다.

# result2 = model2.evaluate(x_test, y_test, batch_size=8)
# print('로드모델_loss: ', result2[0])
# print('로드모델_acc: ', result2[1])

# w 값이 같아서 위와 같은 값이 나온다.
# 로드모델_loss:  0.05850084125995636
# 로드모델_acc:  0.982200026512146

#---------------------------------------------------------

model = load_model('../data/modelcheckpoint/k52_1_mnist_checkpoint.hdf5')
result = model.evaluate(x_test, y_test, batch_size=8)
print('로드체크포인트_loss: ', result[0])
print('로드체크포인트_acc: ', result[1])

# 모델까지 저장되어 있어서 모델이 없어도 돌아가는 것을 알 수 있다.
# 체크포인트는 loss가 가장 낮은 지점이기 때문에 예측값이 더 좋다.
# 위 2개의 로드들은 얼리스탑핑으리 통해 최적의 지점을 지났기 때문이다.
# 로드체크포인트_loss:  0.045716542750597
# 로드체크포인트_acc:  0.9853000044822693

