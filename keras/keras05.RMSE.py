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
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일하고 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.3)

#4.평가, 예측
results = model.evaluate(x_test, y_test, batch_size=1)
print('mse, mae: ', results)

y_predict = model.predict(x_test)
#print('y_pred: ', y_predict)

#사이킷런을 써보자(라이브러리임), 그리고 mse랑 mae는 vs에 저장되어있는데 rmse는 없어서 지금 함수를 정의해주는거다.
#그리고 구지 사이킷런에서 불러와서 쓰지 않고 그냥 vs에 저장되어있는 mse를 써도 된다. 이렇게 만들 수도 있다는 것을 알려주는 것이다.
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
#def는 정의하겠다. return 은 값을 주겠다. sqrt는 루트를 씌우는 것
print('RMSE: ', RMSE(y_test, y_predict))
#이렇게 만들어야 앞에 있는 거랑 안 겹치고 어디든 재활용할 수 있다.
print('mse check: ', mean_squared_error(y_test, y_predict))
#이 뒤의 mse와 평가,예측의 mse가 값이 같은지 알라보려고 하는건다. 둘이 아주 근소한 차이가 나지만 비슷하다!
