import numpy as np
from tensorflow.keras.datasets import cifar100   #10가지로 분류하는것. cifar100은 100개로 분류

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)

#전처리 // 3) y벡터화 / 2) x minmax / 1) x traintest 분리

# print(np.min(x_train), np.max(x_train))     #0 255
# print(np.min(y_train), np.max(y_train))     #0 9

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

x_train = x_train.astype('float32')/255.
x_val = x_val.astype('float32')/255.
x_test = x_test.astype('float32')/255.

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])         #(40000, 3072)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2]*x_val.shape[3])                     #(10000, 3072)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])               #(40000, 3072)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Dense(200, input_shape=(3072,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(160, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='acc', patience=16, mode='max')

model.fit(x_train, y_train, epochs=300, batch_size=64, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=32)
print('loss:' ,loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))


# 44-2 cifar100 CNN
# loss: [3.5911879539489746, 0.2443999946117401]

# 44-3 cifar100 DNN
# loss: [3.560333251953125, 0.16259999573230743]
