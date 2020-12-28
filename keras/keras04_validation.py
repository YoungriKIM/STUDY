import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터를 주자
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
#머신이 훈련할 데이터
x_validation = np.array([6,7,8])
y_validation = np.array([6,7,8])
#머신이 검증할 데이터(머신이 지 혼자 훈련도 하고 검증도 하면 성능이 향샹되겠지?)
#머신이 검증하는 거니까 훈련을 하면서 해야겠져 그래서 훈련(fit)할 때 벨리데이션을 넣어줍시다
x_test = np.array([9,10,11])
y_test = np.array([9,10,11])
#사람이 평가할 데이터


#2. 모델을 구성하자
model = Sequential()
model.add(Dense(5, input_dim = 1, activation='linear'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x_train, y_train, epochs=100, batch_size=1, 
            validation_data=(x_validation, y_validation)
)
#앞으로는 val이 거진 들어갈 것이다. 이렇게 하면 학습할 때 옆에 val 녀석들도 보인다. 또 성능도 좋아졌다.

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)

#result = model.predict([9])
result = model.predict(x_train)
print('result: ', result)

#터미널의 학습 기록을 보면 그냥 loss와 mae 보다 val의 loss와 mae가 더 높다는 것을 알 수 있다. 즉 성과가 더 안좋다.
#그래서 실질적으로 val의 결과값을 더 신뢰한다.(훈련한 걸로 검증을 한 거니까)