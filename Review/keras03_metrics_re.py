from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

x_train= np.array([1,2,3,4,5,6,7,8,9,10])
y_train= np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])

model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mes', optimizer = 'adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)

result = model.predict(x_pred)
print('result: ', result)