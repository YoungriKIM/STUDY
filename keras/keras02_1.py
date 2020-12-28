import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
x_test = np.array([6,7,8])
y_test = np.array([6,7,8])
#train, test를 나누었지만 원래 데이터는 전부 합한거다. 내가 슬라이싱한 것 뿐

model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)

result = model.predict([9])
print('result: ', result)