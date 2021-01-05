#  input_shape /input_length / input_dim 알아보기 위한 파일

#1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print('x.shape: ', x.shape) #(4,3)
print('y.shape: ', y.shape) #(4,)

x = x.reshape(4, 3, 1) 

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM #LSTM이 RNN모델의 레이어다.
model = Sequential()

# model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(LSTM(10, activation='relu', input_length = 3, input_dim = 1)) #이렇게도 위의 코드를 대체할 수 있다.
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련 (분류가 아닌 실수가 나올테고 즉 회기모델이라는 뜻이다.)
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

#평가, 예측
loss = model.evaluate(x, y)
print(loss)

x_pred = np.array([5,6,7])
x_pred = x_pred.reshape(1, 3, 1)

result = model.predict(x_pred)
print(result)

# 0.018204331398010254
# [[7.9421453]]