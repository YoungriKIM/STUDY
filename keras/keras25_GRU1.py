#SimpleRNN을 GRU로 바꿔보자

#1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print('x.shape: ', x.shape) #(4,3)
print('y.shape: ', y.shape) #(4,)

#LSTM에 넣으려면 x를 한개씩 잘라서 작업을 해야하고, 3차원으로 바꿔야한다.
x = x.reshape(4, 3, 1) #리쉐잎해도 원소의 개수는 동일하며 데이터 손실은 없다.

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, SimpleRNN, GRU

model = Sequential()
model.add(GRU(10, activation='relu', input_shape=(3,1))) #LSTM의 디폴트 activation은 tanh이다.
#layer에 LSTM을 쓴다. #input_shape의 (3,1)은 삼차원이라는 뜻이다. 맨앞의 행은 무시되었으니까.
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
gru (GRU)                    (None, 10)                390
_________________________________________________________________
dense (Dense)                (None, 20)                220
_________________________________________________________________
dense_1 (Dense)              (None, 10)                210
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 831
Trainable params: 831
Non-trainable params: 0
_________________________________________________________________
'''



'''
#3. 컴파일, 훈련 (분류가 아닌 실수가 나올테고 즉 회기모델이라는 뜻이다.)
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

#평가, 예측
loss = model.evaluate(x, y)
print(loss)

x_pred = np.array([5,6,7]) #(3,) > 1행 3열이었고 또 3개의 스칼라이기도 했는데 > 한개씩 잘라서 해줄 거다. 또 모델에 넣으려면 3차원으로 만들어야 하니까 리쉐잎해주자
x_pred = x_pred.reshape(1, 3, 1) # 1행 3열이고 1개씩 자를거란 뜻이다.

result = model.predict(x_pred)
print(result)

# 23-1 LSTM
# 0.05928492546081543
# [[7.587268]]

# 24-1 SimpleRNN
# 0.03746616840362549
# [[8.049135]]

# 25-1 GRU
# 0.04621589183807373
# [[7.7086034]]
'''