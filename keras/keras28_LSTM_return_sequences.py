#23-3파일을 가져와서 LSTM을 두개를 만드세요.

import numpy as np

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,0], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

#여기서 스케일링을 하고 싶으면 프레딕트까지 해줘야 한다.
#또 리쉐잎하기 전에 해줘야 한다. 3차원은 스케일링에 들어가지 않기 때문 > Review 폴더의 23-3에 함

print(x.shape, y.shape) #(13, 3) (13,)

# x = x.reshape(13,3,1)
### 앞으로는 위 리쉐잎말고
print(x.shape[0]) # 13
print(x.shape[1]) # 3
x = x.reshape(x.shape[0], x.shape[1], 1) #0번째와 1번째는 안 바꾸고 가장 마지막만 바꿔주겠다. 

x_pred = x_pred.reshape(1, 3, 1)

print(x.shape, y.shape) #(13, 3, 1) (13,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(30, return_sequences=True, activation='relu', input_shape=(3,1)))
# default return_sequences = False
# return_sequences = True(다음 레이어에 3차원으로 넘겨준다) (None, 3, 30) 이렇게
model.add(LSTM(60, activation='relu'))
model.add(Dense(87, activation='relu'))
model.add(Dense(87, activation='relu'))
model.add(Dense(90, activation='relu')) #Dense 넘기기 전에는 3차원으로 넘기면 안되서 return_sequences = False
model.add(Dense(1))

model.summary()
'''_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 30)             3840                 #retun_sequences = True 는 다음 레이에 3차원으로 넘겨준다. LSTM은 3차원 데이터를 필요로 해서.
_________________________________________________________________
lstm_1 (LSTM)                (None, 60)                21840                #아웃풋의 노드수가 인풋딤이 된다.
_________________________________________________________________
dense (Dense)                (None, 87)                5307
_________________________________________________________________
dense_1 (Dense)              (None, 87)                7656
_________________________________________________________________
dense_2 (Dense)              (None, 90)                7920
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 91
=================================================================
Total params: 46,654
Trainable params: 46,654
Non-trainable params: 0
_________________________________________________________________
'''

#컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor = 'loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=3, validation_split=0.2, verbose=2, callbacks=[stop])

#평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=3)
print('loss: ', loss)

y_pred = model.predict(x_pred)
print('y_pred: ', y_pred)

# 23-3 LSTM 1개
# loss:  [0.82417231798172]
# y_pred:  [[80.21291]]

# 28-1 LSTM 2개
# loss:  1.2061457633972168
# y_pred:  [[74.85687]]

# 28-2 LSTM 3개
# loss:  1.1704970598220825
# y_pred:  [[81.24987]]

# 28-2 전체 LSTM : 속도가 많이 느려짐
# loss:  1.3885211944580078
# y_pred:  [[75.38896]]

#결과가 더 안좋아진다. lstm를 통과한 값이 연속적인 데이터가 아니기 때문이다. 통상적으로 안 좋아지는데 간혹 2개롤 좋아지는 경우도 있으니 이것도 튠으로 이해하고 해볼 것.
