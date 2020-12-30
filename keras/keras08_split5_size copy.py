#실습 train_size / test_size에 대해 이해할 것

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.2, shuffle=False)
# 파라미터 값이 1.1로 1이 넘는다. 이럴 때는? > Reduce test_size and/or train_size.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.2, shuffle=False)
# 파라미터 값이 0.9로 1이 되지 않는다. 이럴 때는? > 이건 되는데 0.1 부분의 데이터가 빠진채로 돌아간다.

print(x_train)
print(x_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

loss, mae = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('mae: ', mae)

y_predict = model.predict(x_test)
# print(y_predict)

# shuffle = False
# loss:  0.009067046456038952
# mae:  0.0939842239022255

# shuffle = True
# loss:  0.0029667953494936228
# mae:  0.042984962463378906

# validation_spilt=0.2
# loss:  3.1183585633698385e-06
# mae:  0.0015232444275170565

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)

print('R2: ', R2)

# validation_data = (x_val, y_val)
# loss:  5.633293039863929e-05
# mae:  0.006273508071899414
# RMSE:  0.007505526599421211
# R2:  0.9999999280501828
