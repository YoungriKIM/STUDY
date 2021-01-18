# kodex 모델링

import numpy as np
import pandas as pd

# 코덱스 데이터
kodex = pd.read_csv('../data/csv/kodex_inverse.csv', index_col=0, header=0, encoding='cp949', thousands=',') 

# 삼성 데이터
ss_y = np.load('../data/csv/samsung_y.npy', allow_pickle=True)
# print(ss_y.shape)


# 데이터 순서 역으로
kodex = kodex.iloc[::-1].reset_index(drop=True)

# 코덱스 맞춰서 삼성 자르기(y용_시가)
ss_y = ss_y[-1088:]


#데이터 지정
x = kodex.iloc[86:-2, [0,1,2,3]]
y = ss_y[88:]
x_pred = kodex.iloc[-2:, [0,1,2,3]]

# print(x.shape)         #(1000, 4)
# print(y.shape)       #(1000,)
# print(x_pred.shape)     #(2, 4)

# 전처리: 2) minmax / 1) traintestsplit / 3) x 3차원 변환

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

# x_pred = x_pred.values.reshape(1,-1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

a = 2
x_train = x_train.reshape(int(x_train.shape[0]/a), x_train.shape[1], 1*a)
x_val = x_val.reshape(int(x_val.shape[0]/a), x_val.shape[1], 1*a)
x_test = x_test.reshape(int(x_test.shape[0]/a), x_test.shape[1], 1*a)
x_pred = x_pred.reshape(int(x_pred.shape[0]/a), x_pred.shape[1], 1*a)
              
y_train = y_train.reshape(int(y_train.shape[0]/a),1*a)
y_val = y_val.reshape(int(y_val.shape[0]/a),1*a)
y_test = y_test.reshape(int(y_test.shape[0]/a),1*a)

# np.save('../data/npy/ensemble_data_kodex.npy', arr=[x_train, y_train, x_val, y_val, x_test, y_test, x_pred])


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D, LSTM, GRU, LeakyReLU

model = Sequential()
model.add(Conv1D(filters = 400, kernel_size = 2, strides=1, padding = 'same', input_shape = (x_train.shape[1], x_train.shape[2]), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(200, 2, padding='same'))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))


#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=16, mode='min')

# modelpath = '../data/modelcheckpoint/samsung2_{epoch:02d}-{val_loss:08f}.hdf5'
# check = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=20, batch_size=4, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])#, check])

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=4)
print('mse: ', format(result[0], ','))
print('mae: ', format(result[1], ','))

y_pred = model.predict(x_pred)
print('1/19일 삼성주식 시가는: ', y_pred, '입니다.')


#conv1d
# batch 4,4 mse:  37,099,376.0
# mse:  34,878,004.0
# mse:  33,566,772.0
# mse:  38,221,568.0
# mse:  35,828,064.0


