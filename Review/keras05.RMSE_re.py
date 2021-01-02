from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import numpy as np
from numpy import array

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = array([11,12,13,14,15])
y_test = array([11,12,13,14,15])
x_pred = array([16,17,18])

model = Sequential()
model.add(Dense(3, input_dim=1, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.3)

results = model.evaluate(x_test, y_test, batch_size=1)
print('mse, mae: ', results)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE: ', RMSE(y_test, y_predict))

print('mse check: ', mean_squared_error(y_test, y_predict))