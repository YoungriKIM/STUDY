# predcit1은 액면분할 한 후만 데이터에 넣었음

import numpy as np
import pandas as pd

#1. 데이터 불러옴
df = pd.read_csv('../data/csv/ss_data.csv', index_col=0, header=0, encoding='cp949', thousands=',') 
# print(df)

# 데이터 순서 역으로
df2 = df.iloc[::-1].reset_index(drop=True)
# print(df2)  (2400, 14)

print(df2.info())

#   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   시가      2400 non-null   int64
#  1   고가      2400 non-null   int64
#  2   저가      2400 non-null   int64
#  3   종가      2400 non-null   int64
#  4   등락률     2400 non-null   float64
#  5   거래량     2397 non-null   float64
#  6   금액(백만)  2397 non-null   float64
#  7   신용비     2400 non-null   float64
#  8   개인      2400 non-null   int64
#  9   기관      2400 non-null   int64
#  10  외인(수량)  2400 non-null   int64
#  11  외국계     2400 non-null   int64
#  12  프로그램    2400 non-null   int64
#  13  외인비     2400 non-null   float64


# 상관계수 확인

print(df2.corr())

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=0.6, font='Malgun Gothic', rc={'axes.unicode_minus':False})
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True) 
plt.show()

# 사용 할 칼럼 : 시가0 / 고가1 / 저가2 / 종가3 / 등락률4 / 금액6 / 신용비7 / 기관9


'''
# x, y 데이터 지정
x = df2.iloc[1740:2399, [0,1,2,3,4,6,7,9]]
y = df2.iloc[1741:2400, 3]
x_pred = df2.iloc[2399, [0,1,2,3,4,6,7,9]]

# print(x.shape)      # (659, 8)
# print(y.shape)      # (659,)
# print(x_pred.shape)      # (8,)

# print(x)
# print(y)
# print(x_pred)

# 전처리: 2) minmax / 1) traintestsplit / 3) x 3차원 변환

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

# print(x_train.shape)    #(421, 8)
# print(x_val.shape)      #(421, 8)
# print(x_test.shape)     #(132, 8)

x_pred = x_pred.values.reshape(1,-1)
# print(x_pred.shape)      # (8,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)

# print(x_train.shape)        #(421, 8, 1)
# print(x_val.shape)          #(106, 8, 1)
# print(x_test.shape)         #(132, 8, 1)
# print(x_pred.shape)         #(1, 8, 1)

# np.save('../data/npy/samsung_x_train.npy', arr=x_train)
# np.save('../data/npy/samsung_y_train.npy', arr=y_train)
# np.save('../data/npy/samsung_x_val.npy', arr=x_val)
# np.save('../data/npy/samsung_y_val.npy', arr=y_val)
# np.save('../data/npy/samsung_x_test.npy', arr=x_test)
# np.save('../data/npy/samsung_y_test.npy', arr=y_test)
# np.save('../data/npy/samsung_x_pred.npy', arr=x_pred)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D, LSTM, GRU, LeakyReLU

model = Sequential()
model.add(Conv1D(filters = 304, kernel_size = 7, strides=1, padding = 'same', input_shape = (8,1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, 2, activation='relu'))
model.add(Conv1D(40, 2, activation='relu'))
model.add(Conv1D(40, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# model.summary()


#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=16, mode='min')

# modelpath = '../data/modelcheckpoint/samsung_{epoch:02d}-{val_loss:.4f}.hdf5'
# check = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

model.fit(x_train, y_train, epochs=500, batch_size=1, validation_data=(x_val, y_val), verbose=1, callbacks=[stop]) #, check])


#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=1)
print('mse: ', result[0])
print('mae: ', result[1])

y_pred = model.predict(x_pred)
print('1/14일 삼성주식 종가: ', y_pred)


#========= 기록용
# mse:  1286656.875
# mae:  825.32763671875
# 1/14일 삼성주식 종가:  [[90572.59]]

'''