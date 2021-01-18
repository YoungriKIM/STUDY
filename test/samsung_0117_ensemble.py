import numpy as np

# 데이터 불러오기
x1_train = np.load('../data/npy/ensemble_data_ss.npy', allow_pickle=True)[0]
x1_val = np.load('../data/npy/ensemble_data_ss.npy', allow_pickle=True)[1]
x1_test = np.load('../data/npy/ensemble_data_ss.npy', allow_pickle=True)[2]
x1_pred = np.load('../data/npy/ensemble_data_ss.npy', allow_pickle=True)[3]

x2_train = np.load('../data/npy/ensemble_data_kodex.npy', allow_pickle=True)[0]
y2_train = np.load('../data/npy/ensemble_data_kodex.npy', allow_pickle=True)[1]
x2_val = np.load('../data/npy/ensemble_data_kodex.npy', allow_pickle=True)[2]
y2_val = np.load('../data/npy/ensemble_data_kodex.npy', allow_pickle=True)[3]
x2_test = np.load('../data/npy//ensemble_data_kodex.npy', allow_pickle=True)[4]
y2_test = np.load('../data/npy/ensemble_data_kodex.npy', allow_pickle=True)[5]
x2_pred = np.load('../data/npy/ensemble_data_kodex.npy', allow_pickle=True)[6]

#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D, LSTM, GRU, LeakyReLU, concatenate

#모델1
input1 = Input(shape = (x1_train.shape[1], x1_train.shape[2]))
conv1 = Conv1D(filters = 400, kernel_size = 2, strides=1, padding = 'same', activation='relu')(input1)
pool1 = MaxPooling1D(pool_size=2)(conv1)
conv1 = Conv1D(400, 2, padding='same')(pool1)
conv1 = Conv1D(200, 2, padding='same')(conv1)
conv1 = Conv1D(200, 2, padding='same')(conv1)
pool1 = MaxPooling1D(pool_size=2)(conv1)
flat1 = Flatten()(pool1)
dense1 = Dense(16)(flat1)
dense1 = Dense(16)(dense1)
dense1 = Dense(16)(dense1)
dense1 = Dense(4)(dense1)
dense1 = Dense(4)(dense1)

#모델2
input2 = Input(shape = (x2_train.shape[1], x2_train.shape[2]))
conv2 = Conv1D(filters = 400, kernel_size = 2, strides=1, padding = 'same', activation='relu')(input2)
pool2 = MaxPooling1D(pool_size=2)(conv2)
conv2 = Conv1D(200, 2, padding='same')(pool2)
flat2 = Flatten()(conv2)
dense2 = Dense(32)(flat2)
dense2 = Dense(32)(dense2)
dense2 = Dense(16)(dense2)
dense2 = Dense(16)(dense2)
dense2 = Dense(8)(dense2)
dense2 = Dense(4)(dense2)


#모델 병합
merge1 = concatenate([dense1, dense2])
output1 = Dense(2)(merge1)

model = Model(inputs = [input1, input2], outputs = output1)


#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
stop = EarlyStopping(monitor='val_loss', patience=16, mode='min')

# modelpath = '../data/modelcheckpoint/ss_ensemble_{epoch:02d}-{val_loss:08f}.hdf5'
# check = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
reducelr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5, verbose=1)

hist = model.fit([x1_train, x2_train], y2_train, epochs=100, batch_size=2, validation_data=([x1_val, x2_val], y2_val), verbose=2, callbacks=[stop, reducelr])#, check])

#4. 평가, 예측
result = model.evaluate([x1_test, x2_test], y2_test, batch_size=2)
print('mse: ', format(result[0], ','))
print('mae: ', format(result[1], ','))

y_pred = model.predict([x1_pred, x2_pred])
print('1/18일, 19일 삼성주식 시가는: ', y_pred, '입니다.')

# mse:  16,082,799.0

# reduce_lr 적용
# mse:  2,326,659.25