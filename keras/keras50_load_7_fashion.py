# 저장한 npy 파일을 불러와보자

import numpy as np

x_train = np.load('../data/npy/fashion_x_train.npy')
y_train = np.load('../data/npy/fashion_y_train.npy')
x_test = np.load('../data/npyfashion_x_test.npy')
y_test = np.load('../data/npy/fashion_y_test.npy')

# 전처리 // 4)다중분류 y벡터화 / 1)train,val 분리 / 2)minmaxscaler / 3)차원 리쉐잎

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32')/255.
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2]).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# print(x_train.shape)        (48000, 784)
# print(y_train.shape)        (10000, 784)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Dense(200, input_shape=(784,), activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='acc', patience=16, mode='max')

model.fit(x_train, y_train, epochs=100, batch_size=49, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=49)
print('loss:' ,loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

# 42-2 CNN
# 277
# loss: [0.01650778204202652, 0.9014000296592712]
# y_pred:  [9 2 1 1 0 1 4 6 5 7]
# y_test:  [9 2 1 1 6 1 4 6 5 7]

# 42-3 DNN      DNN이 CNN에 육박하지만 더 좋지는 않다. 속도는 훨씬 빠르다.
# 70
# loss: [0.017586635425686836, 0.8916000127792358]
# y_pred:  [9 2 1 1 6 1 4 6 5 7]
# y_test:  [9 2 1 1 6 1 4 6 5 7]

# 50-7 제대로 돌아가는 것 확인