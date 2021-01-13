# Conv1D로 바꾸시오 그리고 비교하시오

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

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D

model = Sequential()
model.add(Conv1D(filters=48, kernel_size=2, input_shape=(13, 1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(36, 1))
model.add(Conv1D(36, 1))
model.add(Flatten())
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
model.fit(x_train, y_train, epochs=200, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=earlystopping)

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

# 54-6 conv1d
# loss:  [0.061490338295698166, 0.9722222089767456]