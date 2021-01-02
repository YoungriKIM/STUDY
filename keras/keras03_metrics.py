import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터를 주자
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

#2. 모델을 구성하자
model = Sequential()
model.add(Dense(5, input_dim = 1, activation='linear'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#컴파일에 metrics로 에큐러시(정확도)를 추가해주었다. >훈련을 하는 과정에서 그 지표를 하나 더 추가해준 것이다.
#그러면 터미널의 로스의 값안에 0.0 즉 에큐러시도 추가된 것을 볼 수 있다.
#모델 구성을 아무리 좋게해도 에큐러시가 0.0이 나오는 이유는 숫자 1.00001과 1은 다르기 떄문이다. 즉 이 모델에서는 사용하기에 적합하다고 볼 수는 없다.
#대신 값이 1 아니면 0 으로 나오는 식의 모델인 분류 모델이라면 괜찮다.
#model.compile(loss='mse', optimizer='adam', metrics=['mse'])
#이렇게 하면 지표로 mse가 추가되어서 나온다. 그런데 loss도 mse니까 둘의 값이 같겠지.
#메트릭스에는 mae나 mse등 두 개 이상을 사용할 수 있기 때문에 []대괄호를 이용해 리스트로 묶어준다.
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1) #metrics를 추가하기 전에는 이 값으로 mes만 나와서 loss로 했지만 다음 파일에는 이름이 다르다.
print('loss: ', loss)

#result = model.predict([9])
result = model.predict(x_train)
print('result: ', result)

#epochs 93/100 ▶ 99/100 이 될 수록 loss 값이 줄어드는 것을 볼 수 있다. 즉 머신의 예측값이 정답과 가까워 지고 있다는 것을 알 수 있다.