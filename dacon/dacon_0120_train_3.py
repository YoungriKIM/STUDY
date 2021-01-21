# 데이터 스플릿 더 정리/ 트레인 366 > 48
# conv1d > LSTM

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
dataset2 = dataset2.iloc[:-103,:]

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
        tmp_y = aaa[i:i+x_row, x_col:x_col+y_col]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

# print(x, '\n\n', y)
x, y = split_xy(aaa, 48,7,48,2)   # 7일치로 RNN식으로 자름
print(x.shape)                    #(52410, 48, 7)
print(y.shape)                    #(52410, 48, 2)


#===================================================================
# 81개의 test의 데이터 불러와서 합치기
def preprocess_data(data):
    temp = data.copy()
    return temp.iloc[-48:,[1,3,4,5,6,7,8]]

df_test = []

for i in range(81):
    file_path = '../data/csv/dacon1/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp)
    df_test.append(temp)

all_test = pd.concat(df_test).values
all_test = all_test.reshape(81, 48, 7)
# print(all_test.shape)   #(81, 48, 7)


#===================================================================
# 전처리

# 1) 2차원으로 만들어서 트레인테스트분리
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])
all_test = all_test.reshape(all_test.shape[0], all_test.shape[1]*all_test.shape[2])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

# 2) MinMax
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
all_test = scaler.transform(all_test)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# (41928, 336)
# (10482, 336)
# (41928, 96)
# (10482, 96)

# 3) 3차원으로 리쉐잎
num1 = 7
num2 = 2
x_train = x_train.reshape(x_train.shape[0], int(x_train.shape[1]/num1), num1)
x_test = x_test.reshape(x_test.shape[0], int(x_test.shape[1]/num1), num1)
all_test = all_test.reshape(all_test.shape[0], int(all_test.shape[1]/num1), num1)

y_train = y_train.reshape(y_train.shape[0], int(y_train.shape[1]/num2), num2)
y_test = y_test.reshape(y_test.shape[0], int(y_test.shape[1]/num2), num2)

# (41928, 48, 7)
# (10482, 48, 7)
# (41928, 48, 2)
# (10482, 48, 2)




#===================================================================
#모델 구성      > Conv1d
#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input, Flatten, MaxPooling1D, Dropout, Reshape, SimpleRNN, LSTM, LeakyReLU, GRU

# model = Sequential()
# model.add(Conv1D(96, 7, input_shape=(x_train.shape[1], x_train.shape[2]), padding='same', activation='relu'))
# model.add(Conv1D(48, 7, padding='same'))
# model.add(Conv1D(48, 7, padding='same'))
# model.add(Conv1D(48, 7, padding='same'))
# model.add(Flatten())
# model.add(Dense(96))
# model.add(Dense(144))
# model.add(Dense(96))
# model.add(Reshape((48,2)))
# model.add(Dense(2))

# model.summary()


#===================================================================
# 핀볼로스 적용
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.backend import mean, maximum
import os
import glob
import random

def quantile_loss(q, y_true, y_pred):
    err = (y_true-y_pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for q in q_lst:
    model = Sequential()
    model.add(Conv1D(96, 2, input_shape=(x_train.shape[1], x_train.shape[2]), padding='same', activation='relu'))
    model.add(Conv1D(48, 2, padding='same'))
    model.add(Conv1D(48, 2, padding='same'))
    model.add(Flatten())
    model.add(Dense(144))
    model.add(Dense(96))
    model.add(Dense(96))
    model.add(Reshape((48,2)))
    model.add(Dense(2))

    # model.summary()

    model.compile(loss = lambda y_true,y_pred: quantile_loss(q,y_true,y_pred), optimizer='adam', metrics=['mse'])

    stop = EarlyStopping(monitor='val_loss', patience=16, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience=8, factor=0.5)
    filepath = '../data/modelcheckpoint/dacon_train_3_{epoch:02d}-{val_loss:.4f}.hdf5'
    check = ModelCheckpoint(filepath=filepath, monitor = 'valo_loss', save_best_only=True, mode='min')

    hist = model.fit(x_train,y_train,validation_split=0.2,epochs=5,batch_size=7,verbose=1,callbacks=[stop,reduce_lr,check])


#===================================================================
#컴파일, 핏
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# stop = EarlyStopping(monitor='val_loss', patience=20, mode='min')
# filepath = '../data/modelcheckpoint/dacon_train_1_{epoch:02d}-{val_loss:.4f},hdf5'
# check = ModelCheckpoint(filepath=filepath, monitor = 'valo_loss', save_best_only=True, mode='min')
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)

# hist = model.fit(x_train, y_train, epochs=100, batch_size=24, validation_split=0.2, verbose=1, callbacks=[stop, reduce_lr])#,check])


#===================================================================
# 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=7)
print('loss: ', result[0])
print('mae: ', result[1])

y_predict = model.predict(all_test)
# print(y_predict.shape)          #(81, 48, 2)


#===================================================================
# 예측값을 submission에 넣기
sub = pd.read_csv('../data/csv/dacon1/sample_submission.csv')

y_predict = (y_predict.reshape(y_predict.shape[0]*y_predict.shape[1], 2),2)
for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains('Day7'), column_name] = y_predict[:,0]
for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains('Day8'), column_name] = y_predict[:,1]

sub.to_csv('../data/csv/dacon1/submmission_0121_1.csv', index=False)

print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')


#===================================================================
# 예측값으로 그래프 그리기
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pandas import DataFrame

graph = pd.read_csv('../data/csv/dacon1/submmission_0121_1.csv').iloc[:,1]
plot = graph.plot()
plot.set_xlabel("time")
plot.set_ylabel("predict")
plt.title("Predict")

plt.show()


#===================================================================

# y 예측값을 csv로 저장
# df = pd.DataFrame(origin_all)
# df.columns = ['TARGET+1', 'TARGET+2']
# df.to_csv('../data/csv/dacon0119_2.csv', sep=',')



# mse:  122.43357849121094
# mse:  108.73043060302734
# mse:  94.28148651123047
# mse:  28.44871711730957
# mse:  42.21013641357422
# mae:  230.33175659179688
# mae:  260.749755859375

# ing