import numpy as np
from tensorflow.keras.datasets import cifar10   #10가지로 분류하는것. cifar100은 100개로 분류

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

x_train = x_train.astype('float32')/255.
x_val = x_val.astype('float32')/255.
x_test = x_test.astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), strides=1, padding='same', input_shape=(32,32,3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(96, (2,2), padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (2,2), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

modelpath = '../data/modelcheckpoint/k46_2_cifar10_{epoch:02d}-{val_loss:.4f}.hdf5'
mc= ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=100, batch_size=36, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, mc])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=18)
print('loss:' ,loss)

# y_pred = model.predict(x_test[:10])
# print('y_pred: ', y_pred.argmax(axis=1))
# print('y_test: ', y_test[:10].argmax(axis=1))

#시각화
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 10

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


#=============================================
# 43-2 cifar CNN
# loss: [0.040339864790439606, 0.7218999862670898]
# y_pred:  [5 8 0 0 6 6 1 6 3 1]
# y_test:  [3 8 8 0 6 6 1 6 3 1]
