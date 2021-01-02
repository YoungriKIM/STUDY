#다:1 mlp

import numpy as np

x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array(range(711, 811))

print(x.shape) #(3,100)
print(y.shape) #(100,)

x = np.transpose(x)
# print(x)
print(x.shape) #(100,3)

from sklearn.model_selection import train_test_split #요건 행을 정리하는 거임, 열을 건드리지 않음 특성(열,칼럼)을 정리하는 거임
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
#그냥 셔플을 돌려버리면 매번 돌릴 때 마다 달라져 비교평가하기가 힘드니까 random난수를 정해주는 것이다.
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

print(x_train.shape)   #(64,3)
print(x_val.shape)   #(16,3)
print(x_test.shape)   #(20,3)

print(y_train.shape)   #(64,)
print(y_val.shape)   #(16,)
print(y_test.shape)   #(20,)


#모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

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

#모델을 대충 짰는데도 결과값이 잘 나오는 이유는 x1, x2, x3에 대한 w가 모두 1(b는 다를 수 있으나 큰 영향 없음)이기 때문이다.