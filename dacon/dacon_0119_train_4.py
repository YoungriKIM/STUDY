# dacon 0118-1 기져와서 씀
# y 데이터 스플릿 부분만 다시
#       * 테스트 넣을 때 파일의 6일꺼만 넣어도 가능
# RNN

import numpy as np
import pandas as pd

dataset = pd.read_csv('../data/csv/dacon1/train/train.csv', index_col=None, header=0)
# print(dataset.shape)
dataset = dataset.iloc[:,[1,3,4,5,6,7,8]]

# print(dataset.shape)      #(52560, 7)
# print(dataset.info())

# 다음날, 다다음날의 TARGET을 오른쪽 열으로 붙임
df1 = dataset['TARGET'].shift(-48)  #다음날
df2 = dataset['TARGET'].shift(-48*2)    #다다음날

dataset2 = pd.concat([dataset, df1, df2], axis=1)
dataset2.columns = ['Hour', 'DHI', 'DNI', 'WS', 'RH','T','TARGET','TARGET+1','TARGET+2']
dataset2 = dataset2.iloc[:-100,:]

# print(dataset2.info())
# print(dataset2.shape)       #(52460, 9)       # 7개는 기준일, 오른쪽으로 1개는 다음 1개는 다다음


aaa = dataset2.values
# print(len(aaa))

def split_xy(aaa, x_row, x_col, y_row, y_col):
    x, y = list(), list()
    for i in range(len(aaa)):
        if i > len(aaa)-x_row:
            break
        tmp_x = aaa[i:i+x_row, :x_col]
        tmp_y = aaa[i+x_row-1, x_col:x_col+y_col]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

# print(x, '\n\n', y)
x, y = split_xy(aaa, 336,7,336,2)   # 7일치로 RNN식으로 자름
# print(x.shape)                    #(52129, 336, 7)
# print(y.shape)                    #(52129, 2)   > (52129, 1, 2)로 바꿔야 함


#===================================================================
# 전처리

# 1) 2차원으로 만들어서 트레인테스트분리
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

# 2) MinMax
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape)        #(41703, 2352)
# print(x_test.shape)         #(10426, 2352)
# print(y_train.shape)        #(41703, 2)   
# print(y_test.shape)         #(10426, 2) 

# 3) x는 4차원 y는 3차원
num1 = [2780,15,112,21]     #for x train
num2 = [695,15,112,21]       #for x test
num3 = [2780,15,2]          #for y train
num4 = [695,15,2]           #for y test

# x_train = x_train.reshape(num1[0], num1[1], num1[2], num1[3])
# x_test = x_test.reshape(num2[0], num2[1], num2[2], num2[3])

y_train = y_train.reshape(num3[0], num3[1], num3[2])
y_test = y_test.reshape(num4[0], num4[1], num4[2])

# 3) x 3차원, y 3차원
num5 = [2780,1680,21]
num6 = [695,1680,21]

x_train = x_train.reshape(num5[0], num5[1], num5[2])
x_test = x_test.reshape(num6[0], num6[1], num6[2])


#===================================================================
#모델 구성      > Conv12d
#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, MaxPooling1D, Dropout, Reshape, Conv2D, MaxPool2D,LeakyReLU, GRU,SimpleRNN

model = Sequential()
model.add(SimpleRNN(7, input_shape=(x_train.shape[1], x_train.shape[2]),activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, 2, padding='same'))
# model.add(Conv2D(64, 2, padding='same'))
# model.add(Conv2D(32, 2, padding='same'))
model.add(Flatten())
model.add(Dense(30))
model.add(Reshape((15,2)))
model.add(Dense(2))
model.add(LeakyReLU())

# model.summary()


#===================================================================
#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
stop = EarlyStopping(monitor='val_loss', patience=20, mode='min')
filepath = '../data/modelcheckpoint/dacon_train_1_{epoch:02d}-{val_loss:.4f},hdf5'
check = ModelCheckpoint(filepath=filepath, monitor = 'valo_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)

hist = model.fit(x_train, y_train, epochs=1, batch_size=48, validation_split=0.2, verbose=1, callbacks=[stop, reduce_lr])#,check])


#===================================================================
#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=48)
print('mse: ', result[0])
print('mae: ', result[1])

print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')

y_predict = model.predict(x_train)
print(y_predict[1,:10,:])

# y = y.reshape(y.shape[0], 1, y.shape[1])
# print(y.shape)                    #(52129, 1, 2)

# x,y 쉐잎 아래로 변환
# (677,11,7,2352)
# (677,11,7,2)


#===================================================================
# y 예측값을 csv로 저장

# y_predict = y_predict.reshape(y_predict.shape[0]*y_predict.shape[1], y_predict.shape[2])

# df = pd.DataFrame(y_predict)
# df2 = df.iloc[-48:, :]
# df2.columns = ['TARGET+1', 'TARGET+2']
# df2.to_csv('../data/csv/dacon0120_1.csv', sep=',')

# print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')


#===================================================================

from tensorflow.keras.backend import mean, maximum

def quantile_loss(q, y_test, y_predict):
  err = (y_test-y_predict)
  return mean(maximum(q*err, (q-1)*err), axis=-1)


model.compile(loss=lambda y_test,y_predict: quantile_loss(0.5,y_test,y_predict))#, **param)

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for q in q_lst:
  model.add(Dense(10))
  model.add(Dense(1))   
  model.compile(loss=lambda y,pred: quantile_loss(q,y,pred), optimizer='adam')
  model.fit(x_train,y_train)

