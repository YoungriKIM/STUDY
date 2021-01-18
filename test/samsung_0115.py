# 01/19 삼성전자 시가 예측하기 / 삼성모델과 인버스모델 앙상블하여 예측 / 병합한 컬럼의 합 6개 이상
# 01/18 AM09:00까지 제출 (첨부: 1.py / 삼성.npy, 인버스.npy / 4.h5 or hdf5)

# 1/19일 삼성시가 모델링

import numpy as np
import pandas as pd

#1. 데이터 불러옴
df = pd.read_csv('../data/csv/ss_data.csv', index_col=0, header=0, encoding='cp949', thousands=',') 
# print(df)

# 데이터 순서 역으로
df2 = df.iloc[::-1].reset_index(drop=True)
# print(df2)  (2400, 14)

# 사용 할 칼럼 : 시가0 / 고가1 / 저가2 / 종가3 / 기관9 / 외인(수량)10 / 외국계11 / 프로그램12

df2 = df2.where(pd.notnull(df2), df2.mean(), axis='columns')     # 결측치에 변수의 평균으로 대체

df3 = df2.iloc[0:1739, [0,1,2,3]]/50        # 액면 분할 전에 /50해서 데이터 합치기
df4 = df2.iloc[1739:-1, [0,1,2,3]]
df5 = pd.concat([df3, df4])
df6 = df2.iloc[0:1739, [9, 10, 11, 12]]
df7 = df2.iloc[1739:-1, [9, 10, 11, 12]]
df8 = pd.concat([df6, df7])
df9 = pd.concat([df5, df8], axis=1)

# print(df9.info())       #2400 > 2399로 1/13일 데이터 삭제 > 1/13,14,15 합쳐서 2402개로 수정할 예정
# 사용 할 칼럼 : 시가0 / 고가1 / 저가2 / 종가3 / 기관11 / 외인(수량)12 / 외국계13 / 프로그램14

# 1/14~15 붙이기
dfadd = pd.read_csv('../data/csv/ss_data_3.csv', index_col=0, header=0, encoding='cp949', thousands=',') 

# 데이터 순서 역으로
dfadd = dfadd.iloc[::-1].reset_index(drop=True)
dfadd_data = dfadd.iloc[-3:, [0,1,2,3,11,12,13,14]]
df0115 = pd.concat([df9, dfadd_data]).reset_index(drop=True)

# print(df0115.shape)            #(2402, 8)
# print(df0115.info())


#코덱스에 넣을 y.npy 저장
# samsung_y =  df0115.iloc[:, 0]
# np.save('/content/drive/My Drive/colab_data/samsung_y.npy', arr=samsung_y)

# x, y 데이터 지정
# 액면분할 시점 : 1740이후 부터
x = df0115.iloc[1400:-2, [0,1,2,3]]
y = df0115.iloc[1402:, 0]
x_pred = df0115.iloc[-2:, [0,1,2,3]]
print(x.shape)                  #(2400, 8)
print(y.shape)                  #(2400,)
print(x_pred.shape)             #(2, 8)


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

# print(x_train.shape)        #(768, 8, 2)
# print(x_val.shape)          #(192, 8, 2)
# print(x_test.shape)         #(240, 8, 2)
# print(x_pred.shape)         #(1, 8, 2)

y_train = np.reshape(y_train.values, (-1, 1))           
y_val = np.reshape(y_val.values, (-1, 1))               
y_test = np.reshape(y_test.values, (-1, 1))                 

y_train = y_train.reshape(int(y_train.shape[0]/a),1*a)
y_val = y_val.reshape(int(y_val.shape[0]/a),1*a)
y_test = y_test.reshape(int(y_test.shape[0]/a),1*a)

# print(y_train.shape)        #(768, 2)
# print(y_val.shape)          #(192, 2)
# print(y_test.shape)         #(192, 2)


np.save('../data/npy/ensemble_data_ss.npy', arr=[x_train, x_val, x_test, x_pred])


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D, LSTM, GRU, LeakyReLU

model = Sequential()
model.add(Conv1D(filters = 400, kernel_size = 2, strides=1, padding = 'same', input_shape = (x_train.shape[1], x_train.shape[2]), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(400, 2, padding='same'))
model.add(Conv1D(200, 2, padding='same'))
model.add(Conv1D(200, 2, padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(4))
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


# batch4,4 mse:  896,363.3125
# mse:  29,824,056.0    1400~
# 600~ mse:  2,358,914.5
# 1800~mse:  40,641,032.0
# mse:  3,595,451.25
# mse:  2,449,047.75
# mse:  1,619,367.625
# mse:  1,420,460.625  4칼럼
# mse:  2,533,156.25 6칼럼
# 8칼럼 2,546,677.33 8칼럼
