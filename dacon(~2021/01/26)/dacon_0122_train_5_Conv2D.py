# 0122-1 번 가져와서 hour, RH 빼고 돌리기

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input, Flatten, MaxPooling1D, Dropout, Reshape, SimpleRNN, LSTM, LeakyReLU, GRU, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.backend import mean, maximum
import os
import glob
import random
import tensorflow.keras.backend as K

#===================================================================
# 데이터셋 불러오기
x_train = np.load('../data/npy/dacon1/0122_5_x.npy', allow_pickle=True)[0]
x_val = np.load('../data/npy/dacon1/0122_5_x.npy', allow_pickle=True)[1]
x_test = np.load('../data/npy/dacon1/0122_5_x.npy', allow_pickle=True)[2]

all_test = np.load('../data/npy/dacon1/0122_5_all_test.npy', allow_pickle=True)

y_train = np.load('../data/npy/dacon1/0122_5_y.npy', allow_pickle=True)[0]
y_val = np.load('../data/npy/dacon1/0122_5_y.npy', allow_pickle=True)[1]
y_test = np.load('../data/npy/dacon1/0122_5_y.npy', allow_pickle=True)[2]

#===================================================================
#퀀타일 로스 적용된 모델 구성 + 컴파일, 훈련까지

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)
# mean = 평균
# K 를 tensorflow의 백앤드에서 불러왔는데 텐서형식의 mean을 쓰겠다는 것이다.

qlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
subfile = pd.read_csv('../data/csv/dacon1/sample_submission.csv')

# model.summary()

for q in qlist:
    patience = 16
    print(str(q)+'번째 훈련')
    modelpath = f'../data/modelcheckpoint/dacon_train_0122_5_{q:.1f}.hdf5'
    model = load_model(modelpath, compile = False)
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam', metrics=['mse'])
    stop = EarlyStopping(monitor ='val_loss', patience=patience, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=patience/2, factor=0.5, verbose=1)
    filepath = f'../data/modelcheckpoint/dacon_train_0122_5_{q:.1f}.hdf5'
    check = ModelCheckpoint(filepath = filepath, monitor = 'val_loss', save_best_only=True, mode='min') #앞에 f를 붙여준 이유: {}안에 변수를 넣어주겠다는 의미
    # hist = model.fit(x_train, y_train, epochs=500, batch_size=48, verbose=1, validation_split=0.2, callbacks=[stop, reduce_lr])#, check])
    
    # 평가, 예측
    result = model.evaluate(x_test, y_test, batch_size=48)
    print('loss: ', result[0])
    print('mae: ', result[1])
    y_predict = model.predict(all_test)
    # print(y_predict.shape)  #(81, 48, 2)
    
    # 예측값을 submission에 넣기
    y_predict = pd.DataFrame(y_predict.reshape(y_predict.shape[0]*y_predict.shape[1],y_predict.shape[2]))
    y_predict2 = pd.concat([y_predict], axis=1)
    y_predict2[y_predict<0] = 0
    y_predict3 = y_predict2.to_numpy()
        
    print(str(q)+'번째 지정')
    subfile.loc[subfile.id.str.contains('Day7'), 'q_' + str(q)] = y_predict3[:,0].round(2)
    subfile.loc[subfile.id.str.contains('Day8'), 'q_' + str(q)] = y_predict3[:,1].round(2)

    # print(subfile.head())

    
subfile.to_csv('../data/csv/dacon1/sub_0122_5.csv', index=False)

#===================================================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')

# split 48 conv1d   loss:  0.8572525382041931
# size 48,1,8 Conv2D    loss:  0.9139912128448486
# size 1,48,8 Conv2D    loss:  0.8800864219665527
# 0121-5 : loss:  0.7130502462387085    0121-5 2.72518
# 섭미션 수정 후 
# ing  loss:  0.7192889451980591    1.92426	 기쁨의 땐스 ^^
# 0122-1 modelcheckpoint 저장해서 씀
# loss:  0.6989001035690308

# 0122-1 에서 Hour, RH 삭제 0122-5
# loss:  0.74512779712677
# 0122-5 modelcheckpoint 저장해서 씀 csv 이름: 0122_5-2
# ing 모델췍 저장중 # loss:  0.7345216870307922
# 모델체크로 돌리고 저장