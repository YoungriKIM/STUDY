import numpy as np
from tensorflow.keras.models import Sequnetial
from tensorflow.keras.layers import Dense
from numpy import array

# x1 = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x1.shape)

x2 = np.array([[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20]])
# print(x2.shape)

x2 = np.transpose(x2)
# print(x2.shape) #(10,2) 칼럼이 2

model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2)

loss, mae = model.evaluate(x, y, baych_size=1)
print('loss: ', loss)
print('mae: ', mae)

y_predict = model.predict(x)
print('y_predict: ', y_predict)