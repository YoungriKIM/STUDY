import numpy as np
import pandas as pd

#1. 데이터 불러옴
df = pd.read_csv('../data/csv/ss_data.csv', index_col=0, header=0, encoding='cp949', thousands=',') 
# print(df)

# 데이터 순서 역으로
df2 = df.iloc[::-1].reset_index(drop=True)
# print(df2)  (2400, 14)


# 상관계수 확인
'''
print(df2.corr())

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=0.6, font='Malgun Gothic', rc={'axes.unicode_minus':False})
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True) 
plt.show()

사용 할 칼럼 : 시가0 / 고가1 / 저가2 / 종가3 / 등락률4 / 금액6 / 신용비7 / 기관9
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

print(x_train.shape)        #(421, 8, 1)
print(x_val.shape)          #(106, 8, 1)
print(x_test.shape)         #(132, 8, 1)
print(x_pred.shape)         #(1, 8, 1)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D

model = Sequential()
model.add(Conv1D(filters = 128, kernel_size = 3, strides=1, padding = 'same', input_shape = (8,1), activation='relu'))
model.add(Flatten())
model.add(Dense(1))


#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=16, mode='min')

model.fit(x_train, y_train, epochs=100, batch_size=4, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])


#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=4)
print('mse: ', result[0])
print('mae: ', result[1])

y_pred = model.predict(x_pred)
print('1/14일 삼성주식 종가: ', y_pred)