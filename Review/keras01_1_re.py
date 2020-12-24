import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train = np.array([1,2,3,4,5,6])
y_train = np.array([2,4,6,8,10,12])
x_test = np.array([7,8,9,10])
y_test = np.array([14,16,18,20])

x_predict = np.array([11,12,13])

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear'))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)

result = model.predict(x_predict)
print('result: ', result)