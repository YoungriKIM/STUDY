# 다:1 mlp - 함수형
# keras10_mlp2를 함수형으로 변형

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array(range(711, 811))

print(x.shape) #(3,100)
print(y.shape) #(100,)

x = np.transpose(x)
# print(x)
print(x.shape) #(100,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

print(x_train.shape)   #(64,3)
print(x_val.shape)   #(16,3)
print(x_test.shape)   #(20,3)

print(y_train.shape)   #(64,)
print(y_val.shape)   #(16,)
print(y_test.shape)   #(20,)


#모델 구성

# model = Sequential()
# model.add(Dense(10, input_dim=3))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(1))

inputs = Input(shape=(3,))
d1 = Dense(10)(inputs)
d2 = Dense(8)(d1)
d3 = Dense(3)(d2)
outputs = Dense(1)(d3)
model = Model(inputs = inputs, outputs = outputs)

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val), verbose=3)

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
