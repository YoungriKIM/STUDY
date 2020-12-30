# 다:다 mlp - 함수형
# keras10_mlp3 를 함수형으로 변형

import numpy as np

x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(711, 811), range(1,101), range(201,301)])

x = np.transpose(x)
y = np.transpose(y)

print(x.shape) #(100,3)
print(y.shape) #(100,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(10, input_dim=3))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(3)) 

fox = Input(shape=(3,))
bear = Dense(5)(fox)
rabbit = Dense(20)(bear)
pig = Dense(15)(rabbit)
bird = Dense(5)(pig)
tiger = Dense(3)(bird)
model = Model(inputs = fox, outputs = tiger)

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2, verbose=3)

loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)
print('mae: ', mae)

y_predict = model.predict(x_test)
# print(y_predict)

#RMSE와 R2 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)