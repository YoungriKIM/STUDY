# CNN으로 구성하시오/ 2차원을 4차원으로 늘려서 하시오.

from sklearn.datasets import load_wine

#1. 데이터 주기
dataset = load_wine()

x = dataset.data
y = dataset.target

# 나누고
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=66)

# 벡터화하고
from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# 범위 0~1사이로
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# print(x_train.shape)            #(113, 13)
# print(x_val.shape)              #(29, 13)
# print(x_test.shape)             #(36, 13)
# print(y_train.shape)            #(113, 3)
# print(y_val.shape)              #(29, 3)
# print(y_test.shape)             #(36, 3)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D

model = Sequential()
model.add(Conv2D(filters = 120, kernel_size=(1,1), padding='same',  activation='relu', input_shape=(13,1,1)))
model.add(Dropout(0.2))
model.add(Conv2D(60, 1))
model.add(Flatten())
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=earlystopping)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=4)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])
print('y_predict_argmax: ', y_predict.argmax(axis=1)) 
print('y_test[-5:-1]_argmax: ', y_test[-5:-1].argmax(axis=1)) 

# 22-3 Dense
# loss:  [0.035107001662254333, 0.9722222089767456]
# y_predict_argmax:  [0 2 0 1]
# y_test[-5:-1]_argmax:  [0 2 0 1]

# 38-5 드랍아웃 적용 (로스가 올랐음 node가 너무 많이 줄어든 모양)
# loss:  [0.1356738656759262, 0.9722222089767456]
# y_predict_argmax:  [0 2 0 1]
# y_test[-5:-1]_argmax:  [0 2 0 1]

# 41-5 wine CNN       로스가 커졌지만 생각처럼 크게 나빠지지 않았음
# 54
# loss:  [0.42325806617736816, 0.9722222089767456]
# y_predict_argmax:  [0 2 0 1]
# y_test[-5:-1]_argmax:  [0 2 0 1]