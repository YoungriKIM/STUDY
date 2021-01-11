# 웹으로 내보내서 더 좋게 보는 것 
# tensorboard로 부르는 법
'''
127.0.0.1 > 내 컴퓨터
:6006 > 주어진 주소

cd \ 
cd study
cd graph
dir/w                           > 그 폴더의 내용이 나옴, 폴더 있는거 확인하고
tensorboard --logdir=. 
                                >>localhost:6006 확인

웹을 켜서 http://127.0.0.1:6006

*할 때마다 그래프 폴더를 지우고 할 것
한 모델씩만 할 수 있음
'''


import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.     
x_test = x_test.reshape(10000, 28, 28, 1)/255.                        

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

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard  #텐서보드

stop = EarlyStopping(monitor='val_loss', patience=5, mode='min') 

# modelpath = './ModelCheckPoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

tb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
#'graph' 도 된다.
#텐서보드의 파라미터 정리하기

hist = model.fit(x_train, y_train, epochs=100, batch_size=28, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, tb])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=28)
print('loss, acc: ', loss, acc)

y_pred = model.predict(x_test[:10])

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
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)            #2행 2열 중 두 번쨰 그래프
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')    #loss로 그리고 점형태로 찍을거고 색은 레드고 라벨은 로스
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()

plt.title('Accuracy')       #정확도
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()



#===================
# 기록용
# 40-2 mnist CNN
# loss, acc:  0.0900002047419548 0.90000319480896        21
# loss, acc:  0.010415063239634037 0.9835000038146973     17
# loss, acc:  0.009324220940470695 0.9854999780654907     69
# # y_pred:  [7 2 1 0 4 1 4 9 5 9]
# y_test:  [7 2 1 0 4 1 4 9 5 9]
