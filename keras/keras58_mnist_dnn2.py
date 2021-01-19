# 데이콘을 위해 4차원을 dnn으로 구성하기 다시 연습해보자 > Dense로 4차원을 받고 플랫튼도 가능하고 다차원으로 출력할 수도 있다.
# 45번을 복사해서

import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.      
x_test = x_test.reshape(10000, 28, 28, 1)/255.    
 
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.5, shuffle=True, random_state=311)

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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Reshape

model = Sequential()
# model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.5))
# model.add(Conv2D(1, 2, strides=1))
# model.add(Conv2D(2, 2, strides=1))
# model.add(Flatten())
# model.add(Dense(1000))
# model.add(Dense(10, activation='softmax'))

model.add(Dense(64, input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# ReduceLROnPlateau 러닝레이트를 줄여보자

modelpath = '../data/modelcheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
stop = EarlyStopping(monitor='val_loss', patience=10, mode='min') 
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
# 기준을 val_loss로 해서 3번 참는데도 개선 없으면 0.5만큼 줄이겠다. 
# 로그 기록
# Epoch 00022: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257. > adam의 디폴트는 0.001이라는 뜻

hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, mc, reduce_lr])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss, acc: ', loss, acc)

y_pred = model.predict(x_test[:10])
# print('y_pred: ', y_pred)


# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))  

plt.subplot(2, 1, 1)         
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')  
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()      

plt.title('Cost Loss')     
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')    

plt.subplot(2, 1, 2)    
plt.plot(hist.history['acc'], marker='.', c='red')  
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('Accuracy')       #정확도
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()


#===================
# 기록용 45
# 40-2 mnist CNN
# loss, acc:  0.0900002047419548 0.90000319480896        21
# loss, acc:  0.010415063239634037 0.9835000038146973     17
# loss, acc:  0.009324220940470695 0.9854999780654907     69

