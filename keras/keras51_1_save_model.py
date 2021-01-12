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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=120, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(100, 2, strides=1))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(80, 2, strides=1))
model.add(Flatten())
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))

# 모델 세이브를 하자. 모델이 끝난 지점에서 하면 모델만 저장된다.
model.save('../data/h5/k51_1_model1.h5')


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

modelpath = '../data/modelcheckpoint/k51_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# d =  정수형으로 10의 자리까지 /f = float 실수형으로 소수 4번째까지 하겠다. ##이부분 찾아보기

stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=30, batch_size=28, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, mc])

# 모델 세이브를 컴파일, 훈련 뒤에 하면 w값까지 저장된다.
model.save('../data/h5/k51_1_model2.h5')



#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=28)
print('loss, acc: ', loss, acc)

y_pred = model.predict(x_test[:10])

# print('y_pred: ', y_pred.argmax(axis=1))
# print('y_test: ', y_test[:10].argmax(axis=1))

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))    

plt.subplot(2, 1, 1)            #2행 1열 중 첫 번쨰 그래프
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')    #loss로 그리고 점형태로 찍을거고 색은 레드고 라벨은 로스
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()      #바탕을 모눈종이 형태로 표현하겠다

plt.title('Cost Loss')       # 손실비용
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')       # loc = location / upper right = 오른쪽 위에 / 내가 명시해준 라벨을 표시해줌

plt.subplot(2, 1, 2)            #2행 2열 중 두 번쨰 그래프
plt.plot(hist.history['acc'], marker='.', c='red')    #loss로 그리고 점형태로 찍을거고 색은 레드고 라벨은 로스
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('Accuracy')       #정확도
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
 # 위랑 비교했을 때 이렇게 레전드에 직접 이름을 넣어줄 수 있다. 또한 위치를 지정하지 않으면 알아서 비어있는 공간으로 들어간다.

plt.show()

# val_loss의 그래프를 더 신뢰해야 한다. 
# matplotlib에 한글 적용할 것



#===================
# 기록용
# 40-2 mnist CNN
# loss, acc:  0.0900002047419548 0.90000319480896        21
# loss, acc:  0.010415063239634037 0.9835000038146973     17
# loss, acc:  0.009324220940470695 0.9854999780654907     69
# # y_pred:  [7 2 1 0 4 1 4 9 5 9]
# y_test:  [7 2 1 0 4 1 4 9 5 9]
