import numpy as np

#1. 데이터
a =np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa =[]
    for i in range(len(seq) - size + 1):
        subset = seq[i :(i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset.shape) #(6,5)

x = dataset[:, :-1]
y = dataset[:, -1]

print(x.shape) #(6,4)
print(y.shape) #(6,)

#2. 모델
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8 , shuffle=True, random_state=311)

x = x.reshape(x.shape[0], x.shape[1], 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

model= load_model('./model/save_keras35.h5')
model.add(Dense(5, name = 'new1'))
model.add(Dense(1, name = 'new2'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=4, validation_split=0.2, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=4)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)

# loss:  [0.3360520601272583, 0.43158674240112305]
# RMSE:  0.5796999669581155
# R2:  0.6639479483087598