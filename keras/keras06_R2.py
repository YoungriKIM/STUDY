from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import numpy as np
from numpy import array

#1. 데이터준다
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = array([11,12,13,14,15])
y_test = array([11,12,13,14,15])
x_pred = array([16,17,18])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(100))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일하고 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=120, batch_size=1, validation_split=0.2)

#4.평가, 예측
results = model.evaluate(x_test, y_test, batch_size=1)
print('mse, mae: ', results)

y_predict = model.predict(x_test)
#print('y_pred: ', y_predict)

#사이킷런을 써보자(라이브러리임), 그리고 mse랑 mae는 vs에 저장되어있는데 rmse는 없어서 지금 함수를 정의해주는거다.
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

print('mse check: ', mean_squared_error(y_test, y_predict))

#r2를 해보자! (r2는 1.0에 가까우면 좋은거고 0이면 안좋은거다. 이번에도 사이킷런에서 가져와서 써보자)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)