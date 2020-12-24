#전체 공간 주석은 맨 윗줄에 '''하고 맨 아래줄에 '''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import numpy as np
from numpy import array

#1. 데이터
#x = np.array(range(1,101))
#이러면 1이상부터 101미만까지 즉 100개가 나온다
#x = np.array(range(100))
#이러면 0(시작은 늘 0부터다)에서 99까지 100개가 나온다.
x = np.array(range(1,101))
y = np.array(range(101,201)) #이러면 w는 1이고 b가 +100이 되는 수식이다.

x_train = x[:60] #이러면 그냥 숫자 0~59이라는게 아니라 위에 있는 x의 0번째 숫자(=1) 이고 x의 59번째 숫자(=60)이니까 == 1~60 이라는 뜻이다.
x_val = x[60:80] #61~80
x_test = x[80:] #81~100
#이 과정을 라스트의 슬라이싱이라고 한다.

y_train = y[:60] #이러면 그냥 숫자 0~59이라는게 아니라 위에 있는 x의 0번째 숫자(=1) 이고 x의 59번째 숫자(=60)이니까 == 1~60 이라는 뜻이다.
y_val = y[60:80] #61~80
y_test = y[80:] #81~100


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
model.fit(x_train, y_train, epochs=120, batch_size=1, validation_data=(x_val, y_val))

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
