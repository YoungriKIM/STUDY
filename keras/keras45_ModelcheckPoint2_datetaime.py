# 45번 가져와서 모델체크포인트에 저장 된 시간을 같이 저장 할 것임

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

# model.summary()


# 컴파일, 훈련 전에 불러오는 애들
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime   # 시간을 같이 저장하기 전에 불러오는 것

date_now = datetime.datetime.now() 
# print(date_now) #2021-01-27 10:08:05.594710

# 너무 기니까 필요한 것만 프린트하자
date_time = date_now.strftime('%m%d_%H%M') #month day Hour Minute
# print(date_time)

filepath = '../data/modelcheckpoint/'
filename = '_{epoch:02d}-{val_loss:.4f}.hdf5'
modelpath = ''.join([filepath, 'k45_', date_time, filename])
print(modelpath)    # ../data/modelcheckpoint/k45_0127_1018_{epoch:02d}-{val_loss:.4f}.hdf5 의 형식으로 저장된다.


# modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'


stop = EarlyStopping(monitor='val_loss', patience=10, mode='min') #monitor=val_loss도 가능, 그냥 loss보다 val_loss를 신뢰하기도 한다.
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# 파일패스는 파일형태로 기록한다는 뜻이다. 최저점마다 파일을 생성하는데 파일을 만들어 주는 이유는 그 지점의 w를 저장하기 떄문이다. 
# w가 저장되어 있으면 훈련을 다시 할 필요가 없으니까
# 마지막 파일이 가장 좋으니 나머지는 삭제하고 마지막 파일은 가중치를 불러와서 쓸 수 있다.
# k45_mnist_15-0.0554 > 45번 파일의 15번째 훈련이 val_loss가 0.0554로 가장 낮다. 
# 모델이 완벽하다는 가정하에 최적의 기록이다. 튜닝을 확실히 해놓을 것

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, batch_size=28, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, mc])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=28)
print('loss, acc: ', loss, acc)

y_pred = model.predict(x_test[:10])

# print('y_pred: ', y_pred.argmax(axis=1))
# print('y_test: ', y_test[:10].argmax(axis=1))

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))     #10,6정도의 면적을 잡아줌. 구글링해서 정리

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
