#1:다 mlp 함수형

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

x = np.array([range(50)])
y = np.array([range(750,800), range(50,100),range(150,200)])
x_pred2 = np.array([[100]])

#지금은 x.shape가 (1,50), y.shape 가(3,50)이니까 행열을 바꿔주자 (x_pred.shape는 1,1)

x = np.transpose(x)
y = np.transpose(y)

#트레인 테스트 나눠 주고 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

#모델 구성
inputs = Input(shape=(1,))
dense1 = Dense(8,activation='relu')(inputs)
dense1 = Dense(16)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(40)(dense1)
outputs = Dense(3)(dense1)
model = Model(inputs=inputs, outputs=outputs)

#컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=3)

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ',loss)
print('mae: ', mae)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))

print('RMSE: ',RMSE)

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_pred)

print ('R2: ', R2)


y_pred2 = model.predict(x_pred2)
print('y_pred2: ', y_pred2)