from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

x= np.array(range(1,101))
y= np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)
print('mae: ', mae)

y_predict = model.predict(x_test)
#print('y_predict: ', y_predict)