
import numpy as np

# # 데이터 불러오기 main
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

from tensorflow.keras.models import load_model
model = load_model('../data/modelcheckpoint/ss_ensemble_33-18186410.000000.hdf5')

#4. 평가, 예측
result = model.evaluate([x1_test, x2_test], y2_test, batch_size=2)
print('mse: ', format(result[0], ','))
print('mae: ', format(result[1], ','))

y_pred = model.predict([x1_pred, x2_pred])
print('1/18일, 19일 삼성주식 시가는: ', y_pred, '입니다.')

# mse:  2,158,383.0
# mae:  1,120.3511962890625
# 1/18일, 19일 삼성주식 시가는:  [[86808.78 89137.08]] 입니다.