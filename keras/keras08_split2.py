from tensorflow.keras.models import Sequential #이건 모델
from tensorflow.keras.layers import Dense #이건 레이어의 종류
import numpy as np

#1. 데이터
x = np.array(range(1, 101))
y = np.array(range(1, 101))

#앞으로 머신러닝, 데이터 전처리 할 때 사이킷런을 쓸거다
'''
x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_val = y[60:80] 
y_test = y[80:]
이렇게 안하고 쓸 수 있는 코드 없을까? 그게 사이킷런의 train_test_split
'''

from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6) #테스트에 0.4라고 하지 않아도 알아서 나눠짐
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True) #이렇게 셔플을 False하면 순서대로 나온다. 
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, shuffle=True) #섞고싶으면 TRUE. 그럼 셔플은 트루가 디폴트값이군
print(x_train) #이렇게 하니까 무작위로 60개가 나온다. 무작위가 좋은가 순서대로가 좋은가는?: 무작위다. 그래야 마구 섞여있는 트레인과 테스트 데이터의 범위가 비슷해지니까.
print(x_train.shape) #나뉜 것이 몇개인지 확인해보자. 그럼 프린트 되는게 (60,)<-스칼라가 60개. 1차원이라는 뜻임 *input_dim=1 이기 때문에
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential()
# model.add(Dense(1, input_dim=1)) #이것도 된다. 히든이 없을 뿐 머신러닝은 이런 모델 구성을 보인다.
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100) #배치사이즈 안 써도 돌아간다. 디폴트는 32이다.

loss, mae = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('mae: ', mae)

y_predict = model.predict(x_test) #x_test를 넣으면 y_test와 근사한 y_predict가 나온다.
print(y_predict)

# shuffle = False
# loss:  0.009067046456038952
# mae:  0.0939842239022255

# shuffle = True   #섞는게 더 좋지?
# loss:  0.0029667953494936228
# mae:  0.042984962463378906