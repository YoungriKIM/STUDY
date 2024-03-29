#실습
#R2를 음수가 아닌 0.5 이하로 줄이기
#1. 레이어는 인풋과 아웃풋 포함 5개 이상 
#2. batch_size = 1
#3. 에포 = 100이상
#4. 데이터조작금지
# 즉 모델을 엉망으로 만들어도 보라는 거다 이렇게 하면 안된다는 걸 배우자ㅋ

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
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(10))
model.add(Dense(500, activation='relu'))
model.add(Dense(10))
model.add(Dense(500, activation='relu'))
model.add(Dense(10))
model.add(Dense(500, activation='relu'))
model.add(Dense(10))
model.add(Dense(500, activation='relu'))
model.add(Dense(10))
model.add(Dense(500, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일하고 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=150, batch_size=1, validation_split=0.3)

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

#r2를 해보자! (r2는 1에 가까우면 좋은거고 0이면 안좋은거다. 이번에도 사이킷런에서 가져와서 써보자)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)
