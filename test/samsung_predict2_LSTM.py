# predcit1은 액면분할 한 후만 데이터에 넣었음
# predict2에 액면분할 하기 전도 넣어보기 > RNN으로 구성


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
df4 = df2.iloc[1738: , [0,1,2,3]]
df5 = pd.concat([df3, df4])
df6 = df2.iloc[0:1737, [9, 10, 11, 12]]
df7 = df2.iloc[1738: , [9, 10, 11, 12]]
df8 = pd.concat([df6, df7])
df9 = pd.concat([df5, df8], axis=1)

# print(df9.info())
# print(df9.head())
# print(df9.tail())
# print(df9.describe())


# x, y 데이터 지정
# 액면분할 시점 : 1740
x = df2.iloc[1800:2399, [0,1,2,3,4,5,6,7]]
y = df2.iloc[1801:2400, 3]
x_pred = df2.iloc[2399, [0,1,2,3,4,5,6,7]]



# 전처리: 2) minmax / 1) traintestsplit / 3) x 3차원 변환

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

# print(x_train.shape)    #(1535, 8)
# print(x_val.shape)      #(384, 8)
# print(x_test.shape)     #(480, 8)

x_pred = x_pred.values.reshape(1,-1)
# print(x_pred.shape)      # (1, 8)


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

# print(x_train.shape)        #(1535, 8, 1)
# print(x_val.shape)          #(384, 8, 1)
# print(x_test.shape)         #(480, 8, 1)
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
model.add(LSTM(16, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', return_sequences=False))
model.add(Dense(1))


#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=16, mode='min')

# modelpath = '../data/modelcheckpoint/samsung_{epoch:02d}-{val_loss:.4f}.hdf5'
# check = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=20, batch_size=8, validation_data=(x_val, y_val), verbose=1, callbacks=[stop]) #, check])


#4. 평가, 예측
#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=2)
print('mse: ', format(result[0], ','))
print('mae: ', format(result[1], ','))

y_pred = model.predict(x_pred)

y_pred2 = np.array(y_pred)          # 리스트를 numpy로
y_pred3 = float(y_pred2)            # 소수형으로
y_pred4 = format(y_pred3, ',')      # 천의자리에 콤마 넣기

print(y_pred4)
print('1/14일 삼성주식 종가는', y_pred4, '입니다.')


'''
# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss & val_loss')
plt.ylabel('loss, val_loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])
plt.show()
'''


# 기록용
# mse:  2593442.75
# mse:  1649276.75
# mse:  1258485.0
# mse:  1285734.875

