#다:1 mlp

import numpy as np
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape)  #결과가 (10,)으로 나오는데 이 뜻은 스칼라가 10개라는 뜻이다.

# x = np.array([[1,2,3,4,5,6,7,8,9,10],
#               [11,12,13,14,15,16,17,18,19,20]])
# print(x.shape) #결과가 (2, 10)으로 나오는데 2행 10열이라는 뜻이다.


#실습 x.shape = (10,2)가 되게 해라
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [11,12,13,14,15,16,17,18,19,20]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x = x.reshape(10,2)
# x = np.swapaxes(x, 0, 1)
# x = np.transpose(x)
# x = x.T   # 내가 실습한 것

#샘이 알려준 것
x = np.transpose(x)

print(x)
print(x.shape)  #(10,2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense #이것도 괜찮지만 위에 꺼보다 조금 느리다

model = Sequential()
model.add(Dense(10, input_dim=2)) #열이 2개니까 이제 인풋딤이 2
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) #위의 x(10,2)로 y를 찾을 것이다.

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2) #각 컬럼별로 20%임, 한 컬럼만 쓰는 것이 아님

loss, mae = model.evaluate(x, y, batch_size=1)
print('loss: ', loss)
print('mae: ', mae)

y_predict = model.predict(x)
# print(y_predict)

##RMSE와 R2 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print('RMSE: ', RMSE(y_test, y_predict))

# print('mse check: ', mean_squared_error(y_test, y_predict))

# from sklearn.metrics import r2_score
# R2 = r2_score(y_test, y_predict)

# print('R2: ', R2)