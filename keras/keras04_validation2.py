from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense #덴스느 y = wx 로 생각해라
import numpy as np
from numpy import array #이러면 앞으로 np.array() 를 array로 써도 되겠지

#1. 데이터준다
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = array([11,12,13,14,15])
y_test = array([11,12,13,14,15])
x_pred = array([16,17,18]) #pred = predict

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일하고 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)
#이때 발리데이션에서 쪼갠다는 것인데, 핏에 넣은 train 데이터의 20%를 쓴다는 것이다. 즉 9,10임 그리고 임의적으로 떼옴

#4.평가, 예측
results = model.evaluate(x_test, y_test, batch_size=1) #이번에는 loss라고 안하고 result라고 하자. 헷갈리니까
print('results: ', results)

y_pred = model.predict(x_pred) #x_pred를 넣어서 나오는 y_pred니까
print('y_pred: ', y_pred)