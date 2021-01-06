#Dense 모델을 구성하시오

import numpy as np

#1. 데이터 제공
a = np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):     #행
        subset = seq[i : (i+size)]           #열
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

x = dataset[:,:4] #[행, 열] = [0:6(모든 행), 0:4] = [:, :4] 
y = dataset[:, 4] #[행, 열] = [0:6(모든 행), 4] = [:, 4] = [:, -1] = [0:-1, -1]
print(x.shape) #(6,4)
print(y.shape) #(6,)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(16, input_shape=(4,), activation='relu'))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=4, verbose=2, validation_split=0.2)

#평가, 예측
loss = model.evaluate(x, y, batch_size=4)
print('loss: ', loss)

x_pred = np.array(range(7,11))
x_pred = x_pred.reshape(1, 4)
y_pred = model.predict(x_pred)
print('y_pred: ', y_pred)

# 32-1 LSTM 모델
# loss:  0.0005612891400232911
# y_pred:  [[4.9478393]]

# 32-2 Dense 모델
# loss:  0.13772261142730713
# y_pred:  [[11.845067]]