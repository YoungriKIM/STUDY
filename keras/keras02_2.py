#이 파일은 keras02_1을 수정한 것임

#처음에 세팅하고
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터를 주자
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,5,8,10,12,14,16,18,20])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([111,112,113,114,115,116,117,118,119,120])

x_predict = np.array([111,112,113]) #예측할 때 쓸 것도 주자

#2. 모델을 구성하자
model = Sequential()
model.add(Dense(50, input_dim = 1, activation='relu'))
model.add(Dense(1))
#이 부분을 바꾸는 걸 튜닝, 튠 한다고 한다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=120, batch_size=10)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)

result = model.predict([x_predict])
print('result: ', result)

#숙제: loss가 0에 가깝게 만들어라
#이 경우에 train과 test 데이터의 패턴과 성질이 다르다. 트레인은 w는 2, 테스트는 w는1에 b는 +10이다. 내가 준 데이터의 양이 작고 상관성이 적기 때문에 나오는 결과 값의 차이가 큰 것이 당연.
#숙제를 낸 의도는 내가 바꿀 수 있는 것들(레이어의 깊이, 갯수, 에포크 횟수, 뱃치값의 수 등)을 바꾸며 감을 익히라는 거다.