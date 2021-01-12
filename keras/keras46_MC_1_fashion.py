# ModelCheckPoint 적용

import numpy as np

#1. 데이터
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(filters = 120, kernel_size = (2,2), strides=1, padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(80, 2))
model.add(Flatten())
model.add(Dense(60, activation='relu'))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

stop = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

modelpath = '../data/modelcheckpoint/k46_1_fashion_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min')

hist = model.fit(x_train, y_train, epochs=10, batch_size=56, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, mc])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=56)
print('loss:' ,loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

#시각화
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 10

plt.figure(figsize=(13,10))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('손실 비용')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()

plt.title('정확도')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.show()

#==============================
# 42-2 CNN
# 277
# loss: [0.01650778204202652, 0.9014000296592712]
# y_pred:  [9 2 1 1 0 1 4 6 5 7]
# y_test:  [9 2 1 1 6 1 4 6 5 7]