from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

loss, mae = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('mae: ', mae)

y_predict = model.predict(x_test)
print(y_predict)

# shuffle = False
# loss:  0.009067046456038952
# mae:  0.0939842239022255

# shuffle = True
# loss:  0.0029667953494936228
# mae:  0.042984962463378906

# validation_spilt=0.2   #좋아지지는 않았네, 무조건 좋아질 거 라고 생각하지 말자 이 경우는 데이터가 적어서 그렇다
# loss:  3.1183585633698385e-06
# mae:  0.0015232444275170565