# Conv1D로 바꾸시오 그리고 비교하시오

import numpy as np

#1 데이터 주고
from tensorflow.keras.datasets import boston_housing
(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

x_train = train_data
y_train = train_target
x_test = test_data
y_test = test_target

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

print(x_train.shape) #(404, 13)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, MaxPooling1D, Flatten, Dropout

input1 = Input(shape=(13,1))
conv1 = Conv1D(260, 1, activation='relu')(input1)
pool1 = MaxPooling1D(pool_size=2)(conv1)
conv1 = Conv1D(130, 1)(pool1)
pool1 = MaxPooling1D(pool_size=2)(conv1)
flat1 = Flatten()(pool1)
dense1 = Dense(30)(flat1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=2000, batch_size=2, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping])

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=2)
print('loss, mae: ', loss, mae)

y_predict = model.predict(x_test)

#RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)




#1번파일
# loss, mae:  11.377790451049805 2.260841131210327
# RMSE:  3.3730985125849697
# R2:  0.8633197073667653

#2번파일 
# loss, mae:  23.424747467041016 3.4580018520355225
# RMSE:  4.8399119673460556
# R2:  0.7186008543791937

# 54-2 Conv1D
# loss, mae:  36.41939163208008 4.536836624145508
# RMSE:  6.034847720372373
# R2:  0.5624975580096943