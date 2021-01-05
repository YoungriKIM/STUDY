import numpy as np

#1. 데이터 제공
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print('x.shape: ', x.shape) #(4, 3)
print('y.shape: ', y.shape) #(4,)

#LSTM에 넣으려면 x가 3차원이 되어야 한다.
x = x.reshape(4, 3, 1)
#4행 3열을 1개씩 자르겠다.

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape=(3,1))
lstm1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(20)(lstm1)
dense1 = Dense(10)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)

model.summary()
'''
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3, 1)]            0
_________________________________________________________________
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 20)                220
_________________________________________________________________
dense_1 (Dense)              (None, 10)                210
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 921
Trainable params: 921
Non-trainable params: 0
_________________________________________________________________
'''

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=100, batch_size=1, verbose=2)

#평가, 예측
loss = model.evaluate(x,y,batch_size=1)
print('loss: ', loss)

x_pred = np.array([5,6,7])
x_pred = x_pred.reshape(1,3,1)

result = model.predict(x_pred)
print(result)

# loss:  0.014041170477867126
# [[8.09698]]