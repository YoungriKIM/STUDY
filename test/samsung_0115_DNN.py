# 0114-1 은 종가 예측
# 0114-2 시가예측 / 15일 데이터 합치기 / y결과 이틀치로

import numpy as np
import pandas as pd

#1. 데이터 불러옴
df = pd.read_csv('../data/csv/ss_data.csv', index_col=0, header=0, encoding='cp949', thousands=',') 
# print(df)

# 데이터 순서 역으로
df2 = df.iloc[::-1].reset_index(drop=True)
# print(df2)  (2400, 14)


# 상관계수 확인

# print(df2.corr())

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(font_scale=0.6, font='Malgun Gothic', rc={'axes.unicode_minus':False})
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True) 
# plt.show()

# 사용 할 칼럼 : 시가0 / 고가1 / 저가2 / 종가3 / 기관9 / 외인(수량)10 / 외국계11 / 프로그램12

df2 = df2.where(pd.notnull(df2), df2.mean(), axis='columns')     # 결측치에 변수의 평균으로 대체

df3 = df2.iloc[0:1737, [0,1,2,3]]/50        # 액면 분할 전에 /50해서 데이터 합치기
df4 = df2.iloc[1738:-1 , [0,1,2,3]]
df5 = pd.concat([df3, df4])
df6 = df2.iloc[0:1737, [9, 10, 11, 12]]
df7 = df2.iloc[1738:-1 , [9, 10, 11, 12]]
df8 = pd.concat([df6, df7])
df9 = pd.concat([df5, df8], axis=1)

# print(df9.info())       #2399 > 2398 1/14일 없앰

####
# 1/14~15 붙이기
dfadd = pd.read_csv('../data/csv/ss_data_3.csv', index_col=0, header=0, encoding='cp949', thousands=',') 

# 데이터 순서 역으로
dfadd = dfadd.iloc[::-1].reset_index(drop=True)
dfadd_data = dfadd.iloc[77:, [0,1,2,3,11,12,13,14]]
df0114 = pd.concat([df9, dfadd_data]).reset_index(drop=True)

# print(df0114.shape)     #(2401, 8)
# print(df0114.tail())
# 사용 할 칼럼 : 시가0 / 고가1 / 저가2 / 종가3 / 기관11 / 외인(수량)12 / 외국계13 / 프로그램14

# x, y 데이터 지정
# 전체데이터 2401 
x = df0114.iloc[1499:2399, [0,1,2,3]]          #(2300, 6)
y = df0114.iloc[1501:2401, 0]                     #(2300,)
x_pred = df0114.iloc[-3:-1, [0,1,2,3]]       #(2, 6)

print(x.shape)
print(y.shape)
print(x_pred.shape)

# 전처리: 2) minmax / 1) traintestsplit / 3) x 3차원 변환

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

y_train = y_train.values.reshape(-1,1)
y_val = y_val.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

a=2
x_train = x_train.reshape(int(x_train.shape[0]/a), x_train.shape[1]*a,1)
x_val = x_val.reshape(int(x_val.shape[0]/a), x_val.shape[1]*a,1)
x_test = x_test.reshape(int(x_test.shape[0]/a), x_test.shape[1]*a,1)
x_pred = x_pred.reshape(int(x_pred.shape[0]/a), x_pred.shape[1]*a,1)

y_train = y_train.reshape(int(y_train.shape[0]/a), 1*a)
y_val = y_val.reshape(int(y_val.shape[0]/a), 1*a)
y_test = y_test.reshape(int(y_test.shape[0]/a), 1*a)

# print(x_train.shape)
# print(x_val.shape)
# print(x_test.shape)
# print(x_pred.shape)
# (736, 12)
# (184, 12)
# (230, 12)
# (1, 12)
# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)
# (736, 2)
# (184, 2)
# (230, 2)

# print('정상이삼^^')

# np.save('../data/npy/samsung3_total.npy', arr=[x_train, y_train, x_val, y_val, x_test, y_test, x_pred])

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D, LSTM, LeakyReLU, GRU, SimpleRNN

model = Sequential()

# model.add(SimpleRNN(16, input_shape=(x_train.shape[1],1), activation='relu', return_sequences=False))
# model.add(GRU(16, input_shape=(x_train.shape[1],1), activation='relu', return_sequences=False))
# model.add(LSTM(16, input_shape=(x_train.shape[1],1), activation='relu', return_sequences=False))

model = Sequential()
model.add(Conv1D(filters = 40, kernel_size = 2, strides=1, padding = 'same', input_shape=(x_train.shape[1],1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(16, 2, padding='same'))
model.add(Conv1D(16, 2, padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(8))
model.add(Dense(8))

model.add(Dense(2))



#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=16, mode='min')

# modelpath = '../data/modelcheckpoint/samsung2_{epoch:02d}-{val_loss:08f}.hdf5'
# check = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=20, batch_size=6, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])#, check])


#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=6)
print('mse: ', format(result[0], ','))
print('mae: ', format(result[1], ','))

y_pred = model.predict(x_pred)
print(y_pred)
# y_pred2 = np.array(y_pred)          # 리스트를 numpy로
# y_pred3 = float(y_pred2)            # 소수형으로
# y_pred4 = format(y_pred3, ',')      # 천의자리에 콤마 넣기

# print('1/15일 삼성주식 종가는', y_pred[1], '입니다.')