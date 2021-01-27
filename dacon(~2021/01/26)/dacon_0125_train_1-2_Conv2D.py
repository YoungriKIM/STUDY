# 0125_1 가져와서 씀
# 아워미닛지표 아워대신 추가
# df["Hour_Minute"] = df["Hour"] * 2 + df["Minute"] // 30

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input, Flatten, MaxPooling1D, Dropout, Reshape, SimpleRNN, LSTM, LeakyReLU, GRU, Conv2D, MaxPool2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.backend import mean, maximum
import os
import glob
import random
import tensorflow.keras.backend as K

#===================================================================
# train 데이터 불러옴

# 원하는 열만 가져오기
dataset = pd.read_csv('../data/csv/dacon1/train/train.csv', index_col=None, header=0)
# print(dataset.shape)
x_train = dataset.iloc[:,[1,2,3,4,5,6,7,8]]
print(x_train.shape)      #(52560, 8)

#===================================================================
# 81개의 all_test 데이터 48개(1일치) 불러와서 합치기
def preprocess_data(data):
    temp = data.copy()
    return temp.iloc[-48:,[1,2,3,4,5,6,7,8]]

df_test = []

for i in range(81):
    file_path = '../data/csv/dacon1/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp)
    df_test.append(temp)

all_test = pd.concat(df_test)
print(all_test.shape)   #(3888, 8)


#===================================================================
# GHI라는 기준 추가
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour'] % 12 - 6)/6 * np.pi/2)
    data['Hour_Minute'] = data["Hour"] * 2 + data["Minute"] // 30
    data.insert(1, 'GHI', data['DNI'] * data['cos'] + data['DHI'])
    data.drop(['cos'], axis=1, inplace = True)
    data.drop(['Hour'], axis=1, inplace = True)
    return data

x_train = Add_features(x_train)     # 트레인에 붙여줌
all_test = Add_features(all_test).values    # 테스트에 붙여줌

print(x_train.shape)      #(52560, 9)   #하나씩 붙은 모습
print(all_test.shape)     #(3888, 9)


#===================================================================
# train에 다음날, 다다음날의 TARGET을 오른쪽 열으로 붙임
day_7 = x_train['TARGET'].shift(-48)      #다음날
day_8 = dataset['TARGET'].shift(-48*2)    #다다음날

x_train = pd.concat([x_train, day_7, day_8], axis=1)
# dataset2.columns = ['Hour', 'GHI', 'DHI', 'DNI', 'WS', 'RH','T','TARGET','TARGET+1','TARGET+2']
x_train = x_train.iloc[:-96,:]  # 마지막 2일은 데이터가 비니까 빼준다

print(x_train.shape)       #(52464, 11)       # 8개는 기준일 +GHI / +day7 + day8

#===================================================================
# x_train을 RNN식으로 데이터 자르기
aaa = x_train.values

def split_xy(aaa, x_row, x_col, y1_row, y1_col, y2_col):
    x, y1, y2 = list(), list(), list()
    for i in range(len(aaa)):
        if i > len(aaa)-x_row:
            break
        tmp_x = aaa[i:i+x_row, :x_col]
        tmp_y1 = aaa[i+x_row-y1_row:i+x_row, x_col:x_col+y1_col]
        tmp_y2 = aaa[i+x_row-y1_row:i+x_row, x_col+y1_col:x_col+y1_col+y2_col]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return np.array(x), np.array(y1), np.array(y2)

# print(x, '\n\n', y)
x_train, y1_train, y2_train = split_xy(aaa, 48,9,48,1,1)     # 30분씩 RNN식으로 자름
# print(x_train.shape)
# print(y1_train.shape)
# print(y2_train.shape)
# (52417, 48, 8)
# (52417, 48, 1)
# (52417, 48, 1)

all_test = all_test.reshape(int(all_test.shape[0]/48), 48, all_test.shape[1])
# print(all_test.shape)     #(81, 48, 9)
all_test = all_test.reshape(all_test.shape[0], all_test.shape[1]*all_test.shape[2])
print(all_test.shape)     #(81, 432)

#===================================================================
# 데이터 전처리 : 준비 된 데이터 x_train / y_train / all_test
# 1) 트레인테스트분리 / 2) 민맥스or스탠다드 / 3) 모델에 넣을 쉐잎

# 1) 2차원으로 만들어서 트레인테스트분리
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
y1_train = y1_train.reshape(y1_train.shape[0], y1_train.shape[1]*y1_train.shape[2])
y2_train = y2_train.reshape(y2_train.shape[0], y2_train.shape[1]*y2_train.shape[2])

from sklearn.model_selection import train_test_split
y1_train, y1_test, y2_train, y2_test = train_test_split(y1_train, y2_train, train_size=0.8, shuffle=True, random_state=311)
from sklearn.model_selection import train_test_split
y1_train, y1_val, y2_train, y2_val = train_test_split(y1_train, y2_train, train_size=0.8, shuffle=True, random_state=311)
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(x_train, train_size=0.8, shuffle=True, random_state=311)
from sklearn.model_selection import train_test_split
x_train, x_val = train_test_split(x_train, train_size=0.8, shuffle=True, random_state=311)

# 2) 스탠다드 스케일러
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
all_test = scaler.transform(all_test)

# 3) 모델에 넣을 쉐잎

# for conv1d & lstm
num1 = 9
num2 = 1
x_train = x_train.reshape(x_train.shape[0], 1, int(x_train.shape[1]/num1), num1)
x_val = x_val.reshape(x_val.shape[0], 1, int(x_val.shape[1]/num1), num1)
x_test = x_test.reshape(x_test.shape[0], 1, int(x_test.shape[1]/num1), num1)
all_test = all_test.reshape(all_test.shape[0], 1, int(all_test.shape[1]/num1), num1)

y1_train = y1_train.reshape(y1_train.shape[0], int(y1_train.shape[1]/num2), num2)
y1_val = y1_val.reshape(y1_val.shape[0], int(y1_val.shape[1]/num2), num2)
y1_test = y1_test.reshape(y1_test.shape[0], int(y1_test.shape[1]/num2), num2)

y2_train = y2_train.reshape(y2_train.shape[0], int(y2_train.shape[1]/num2), num2)
y2_val = y2_val.reshape(y2_val.shape[0], int(y2_val.shape[1]/num2), num2)
y2_test = y2_test.reshape(y2_test.shape[0], int(y2_test.shape[1]/num2), num2)

# print(x_train.shape)        #(33546, 48, 8)
# print(x_val.shape)          #(8387, 48, 8)
# print(x_test.shape)         #(10484, 48, 8)
# print(all_test.shape)       #(81, 48, 8)
# print(y1_train.shape)       #(33546, 48, 1)
# print(y1_val.shape)         #(8387, 48, 1)
# print(y1_test.shape)        #(10484, 48, 1)
# print(y2_train.shape)       #(33546, 48, 1)
# print(y2_val.shape)         #(8387, 48, 1)
# print(y2_test.shape)        #(10484, 48, 1)

#===================================================================
#퀀타일 로스 적용된 모델 구성 + 컴파일, 훈련까지

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)
# mean = 평균
# K 를 tensorflow의 백앤드에서 불러왔는데 텐서형식의 mean을 쓰겠다는 것이다.

qlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
subfile = pd.read_csv('../data/csv/dacon1/sample_submission.csv')


def mymodel():
    model = Sequential()
    model.add(Conv2D(96, (1,2), input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), padding='same', activation='relu'))
    model.add(Conv2D(96, (1,2), padding='same'))
    model.add(Conv2D(96, (1,2), padding='same'))
    model.add(Flatten())
    model.add(Dense(96))
    model.add(Dense(48))
    model.add(Reshape((48,1)))
    model.add(Dense(1))
    return model

# model.summary()

loss_list1 = []
loss_list2 = []

# y1
for q in qlist:
    patience = 8
    print(str(q)+'번째 훈련(y1)')
    model = mymodel()
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam', metrics=['mse'])
    stop = EarlyStopping(monitor ='val_loss', patience=patience, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=patience/2, factor=0.5, verbose=1)
    filepath = f'../data/modelcheckpoint/dacon_train_0125_1_y1_{q:.1f}.hdf5'  #앞에 f를 붙여준 이유: {}안에 변수를 넣어주겠다는 의미
    check = ModelCheckpoint(filepath = filepath, monitor = 'val_loss', save_best_only=True, mode='min')
    hist = model.fit(x_train, y1_train, epochs=1000, batch_size=96, verbose=1, validation_split=0.2, callbacks=[stop, reduce_lr])#, check])
    
    # 평가, 예측
    result = model.evaluate(x_test, y1_test, batch_size=48)
    print('loss, mae: ', result)
    loss_list1.append(result)
    y1_predict = model.predict(all_test)
    # print(y_predict.shape)  #(81, 48, 2)
    
    # 예측값을 submission에 넣기
    y1_predict = pd.DataFrame(y1_predict.reshape(y1_predict.shape[0]*y1_predict.shape[1],y1_predict.shape[2]))
    y1_predict2 = pd.concat([y1_predict], axis=1)
    y1_predict2[y1_predict<0] = 0
    y1_predict3 = y1_predict2.to_numpy()
        
    print(str(q)+'번째 지정')
    subfile.loc[subfile.id.str.contains('Day7'), 'q_' + str(q)] = y1_predict3[:,:].round(2)

# y2
for q in qlist:
    patience = 8
    print(str(q)+'번째 훈련(y2)')
    model = mymodel()
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam', metrics=['mse'])
    stop = EarlyStopping(monitor ='val_loss', patience=patience, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=patience/2, factor=0.5, verbose=1)
    filepath = f'../data/modelcheckpoint/dacon_train_0125_1_y1_{q:.1f}.hdf5'
    check = ModelCheckpoint(filepath = filepath, monitor = 'val_loss', save_best_only=True, mode='min') #앞에 f를 붙여준 이유: {}안에 변수를 넣어주겠다는 의미
    hist = model.fit(x_train, y2_train, epochs=1000, batch_size=96, verbose=1, validation_split=0.2, callbacks=[stop, reduce_lr])#, check])
    
    # 평가, 예측
    result = model.evaluate(x_test, y2_test, batch_size=48)
    print('loss, mae: ', result)
    loss_list2.append(result)
    y2_predict = model.predict(all_test)
    # print(y_predict.shape)  #(81, 48, 2)
    
    # 예측값을 submission에 넣기
    y2_predict = pd.DataFrame(y2_predict.reshape(y2_predict.shape[0]*y2_predict.shape[1],y2_predict.shape[2]))
    y2_predict2 = pd.concat([y2_predict], axis=1)
    y2_predict2[y2_predict<0] = 0
    y2_predict3 = y2_predict2.to_numpy()
        
    print(str(q)+'번째 지정')
    subfile.loc[subfile.id.str.contains('Day8'), 'q_' + str(q)] = y2_predict3[:,:].round(2)

loss_list1 = np.array(loss_list1)
loss_list1 = loss_list1.reshape(9,-1)
loss_list2 = np.array(loss_list2)
loss_list2 = loss_list2.reshape(9,-1)
print('loss1 : \n', loss_list1)
print('loss2 : \n', loss_list2)

subfile.to_csv('../data/csv/dacon1/sub_0125_1_2.csv', index=False)

#===================================================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')

# split 48 conv1d   loss:  0.8572525382041931
# size 48,1,8 Conv2D    loss:  0.9139912128448486
# size 1,48,8 Conv2D    loss:  0.8800864219665527
# 0121-5 : loss:  0.7130502462387085    0121-5 2.72518
# 섭미션 수정 후 
# ing  loss:  0.7192889451980591    1.92426	 기쁨의 땐스 ^^
# first batch:48
# b 24 / 2-1 : loss:  0.7156097292900085 mae:  230.326797485351
# b 8 / 2-2 : loss:  0.6771473288536072 mae:  216.96609497070312   >  2.1257741416	
# b 96 / 2-3 : loss:  0.7318692803382874  mae:  232.2933807373047  > 1.9223068909
# conv1d y 나누고 0125_1-2 : loss:  0.5990279316902161 mae:  209.40272521972656 > 2.4307309863
# conv2d y 나누고 batch48 0125_1-3-1  loss:  0.8756627440452576  mae:  265.82257080078125 > 1.9833854844
# conv2d y 나누고 batch96 0125_1-3-2  파일 확인 >  1.9486018202
# 0125_1-3-2 에서 아워미닛 지표 추가해서 돌리기 0125_1_2 > 1.9467965681
# loss1 :
#  [[  1.25524724 334.57583618]
#  [  1.98180616 185.45890808]
#  [  2.33498478 125.14103699]
#  [  2.45390248 109.40805054]
#  [  2.38803482 117.77754211]
#  [  2.14037037 135.28276062]
#  [  1.75184977 160.40063477]
#  [  1.30525982 192.33322144]
#  [  0.69736141 213.98092651]]
# loss2 :
#  [[  1.37088549 427.3223877 ]
#  [  2.20932174 234.40475464]
#  [  2.62492037 151.09599304]
#  [  2.79542542 127.25587463]
#  [  2.66974282 138.31889343]
#  [  2.36708522 165.735672  ]
#  [  1.917557   191.42324829]
#  [  1.37210178 216.39720154]
#  [  0.76249713 241.17300415]]