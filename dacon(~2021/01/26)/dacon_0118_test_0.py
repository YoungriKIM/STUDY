# dacon 0118-2 파일에 test0 파일을 넣어 예측값 뽑겠음
# conv1d

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
dataset2 = dataset2.iloc[:-96,:]

# print(dataset2.info())
# print(dataset2.shape)       #(52464, 9)       # 7개는 기준일, 오른쪽으로 1개는 다음 1개는 다다음
aaa = dataset2.values
# print(len(aaa))

def split_xy(aaa, x_row, x_col, y_row, y_col):
    x, y = list(), list()
    for i in range(len(aaa)):
        if i > len(aaa)-x_row:
            break
        tmp_x = aaa[i:i+x_row, :x_col]
        tmp_y = aaa[i:i+y_row, x_col:x_col+y_col]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

# print(x, '\n\n', y)
x, y = split_xy(aaa, 336,7,336,2)   # 7일치로 RNN식으로 자름
# print(x.shape)                    #(52129, 336, 7)
# print(y.shape)                    #(52129, 336, 2)

# print(y[1,:,:].shape)

# x_pred 불러오기
dataset_0 = pd.read_csv('../data/csv/dacon1/test/1.csv', index_col=None, header=0)
x_pred = dataset_0.iloc[:, [1,3,4,5,6,7,8]]

print(x_pred.shape) #(336, 7)
x_pred = x_pred.values.reshape(1, x_pred.shape[0]*x_pred.shape[1])

#===================================================================
# 전처리

# 1) 2차원으로 만들어서 트레인테스트분리
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

# 2) MinMax
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

# print(x_train.shape)        #(33362, 2352)
# print(x_val.shape)          #(8341, 2352)
# print(x_test.shape)         #(10426, 2352)
# print(y_train.shape)        #(33362, 672)
# print(y_val.shape)          #(8341, 672)
# print(y_test.shape)         #(10426, 672)

# 3) 3차원으로 리쉐잎
num1 = 7
num2 = 2
x_train = x_train.reshape(x_train.shape[0], int(x_train.shape[1]/num1), num1)
x_val = x_val.reshape(x_val.shape[0], int(x_val.shape[1]/num1), num1)
x_test = x_test.reshape(x_test.shape[0], int(x_test.shape[1]/num1), num1)

x_pred = x_pred.reshape(x_pred.shape[0], int(x_pred.shape[1]/num1), num1)

y_train = y_train.reshape(y_train.shape[0], int(y_train.shape[1]/num2), num2)
y_val = y_val.reshape(y_val.shape[0], int(y_val.shape[1]/num2), num2)
y_test = y_test.reshape(y_test.shape[0], int(y_test.shape[1]/num2), num2)

# print(x_train.shape)        #(33362, 336, 7)
# print(x_val.shape)          #(8341, 336, 7)
# print(x_test.shape)         #(10426, 336, 7)
# print(y_train.shape)        #(33362, 336, 2)
# print(y_val.shape)          #(8341, 336, 2)
# print(y_test.shape)         #(10426, 336, 2)


#===================================================================
#모델 구성      > Conv1d
#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, MaxPooling1D, Dropout, Reshape

model = Sequential()
model.add(Conv1D(96, 7, input_shape=(x_train.shape[1], x_train.shape[2]), padding='same', activation='relu'))
model.add(Conv1D(48, 7, padding='same'))
model.add(Conv1D(48, 7, padding='same'))
model.add(Conv1D(28, 7, padding='same'))
# model.add(Dropout(0.2))
# model.add(Flatten())
model.add(Dense(28))
model.add(Dense(14))
model.add(Dense(7))
model.add(Dense(2))

# model.summary()


#===================================================================
#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
stop = EarlyStopping(monitor='val_loss', patience=20, mode='min')
filepath = '../data/modelcheckpoint/dacon_train_1_{epoch:02d}-{val_loss:.4f},hdf5'
check = ModelCheckpoint(filepath=filepath, monitor = 'valo_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)

hist = model.fit(x_train, y_train, epochs=20, batch_size=48, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, reduce_lr])#,check])


#===================================================================
#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=48)
print('mse: ', result[0])
print('mae: ', result[1])

y_predict = model.predict(x_pred)

# #RMSE와 R2 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print('RMSE: ', RMSE(y_test, y_predict))

'''
#===================================================================
# 이 y 예측값을 RNN방식으로 자르기 전으로 되돌려야 함
print('y_predict.shape:' , y_predict.shape)      #(52129, 336, 2)

print('check1')

bbb = y_predict

# [0,:,:]은 그대로 넣고 1부터 하나씩 가져와서 붙일거임
def back_origin(bbb, b_len, b_row):
    y_result = list()
    for i in range(1,b_len):
        if i > b_len:
            break
        # tmp_origin_top = bbb[0,:,:]
        y_result.append(bbb[i, -1 , :])

        # y_result2 = np.append(tmp_origin_top, y_result, axis=0)

    return np.array(y_result)

print('check2')

origin_2 = back_origin(y_predict, 52129, 336)
# print(origin_2.shape)

origin_1 = bbb[0,:,:]     #(336, 2)

print('check3')

# 0 전부와 1: 를 붙여줌     > 원라는 y의 원래 쉐잎: (52464, 2)
origin_all = np.append(origin_1, origin_2, axis=0)
print(origin_all.shape)     #(52464, 2)

# y 예측값을 csv로 저장
df = pd.DataFrame(origin_all)
df.columns = ['TARGET+1', 'TARGET+2']
df.to_csv('../data/csv/dacon0119_2.csv', sep=',')
'''
y_predict = y_predict.reshape(y_predict.shape[1], y_predict.shape[2])
df = pd.DataFrame(y_predict)
df.columns = ['TARGET+1', 'TARGET+2']
df.to_csv('../data/csv/dacon_test_1.csv', sep=',')

print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')


# mse:  122.43357849121094
# mse:  108.73043060302734
# mse:  94.28148651123047
