import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([range(1,101), range(101, 201), range(201,301), range(301,401), range(401,501)])
y = np.array([range(311, 411), range(323, 423)])
x_pred2 = np.array([1,101,201,301,401])

x = np.transpose(x)
y = np.transpose(y)
x_pred2 = x_pred2.reshape(1, 5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True)

model = Sequential()
model.add(Dense(10, input_dim=5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)
print('mae: ', mae)

y_predict = model.predict(x_test)

y_predict2 = model.predict(x_pred2)
print('y_pred2: ', y_predict2)