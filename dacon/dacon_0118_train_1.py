# 모델을 만들기 위한 파일
# 데이터 전처리, 튜닝은 다음 파일로

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
print(len(aaa))

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
print(x.shape)                    #(52129, 336, 7)
print(y.shape)                    #(52129, 336, 2)

# print(y[1,:,:].shape)


# 전처리
# y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])


#모델 구성      > Conv1d
#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, MaxPooling1D, Dropout, Reshape

model = Sequential()
model.add(Conv1D(5, 48, input_shape=(x.shape[1], x.shape[2]), padding='same', activation='relu'))
model.add(Dropout(0.2))
# model.add(Flatten())
model.add(Dense(2))

model.summary()


#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x, y, epochs=1, batch_size=48, verbose=1)

#4. 평가, 예측
result = model.evaluate(x, y, batch_size=48)
print('mse: ', result[0])
print('mae: ', result[1])

y_predict = model.predict(x)

# #RMSE와 R2 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print('RMSE: ', RMSE(y[:100], y_predict1))

# from sklearn.metrics import r2_score
# R2 = r2_score(y[:100], y_predict1)
# print('R2: ', R2)



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

print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')


# y 예측값을 csv로 저장
# df = pd.DataFrame(origin_all)
# df.columns = ['TARGET+1', 'TARGET+2']
# df.to_csv('../data/csv/dacon0119_2.csv', sep=',')
