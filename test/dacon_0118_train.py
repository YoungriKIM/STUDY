import numpy as np
import pandas as pd

dataset = pd.read_csv('../data/csv/dacon1/train/train.csv', index_col=None, header=0)
# print(dataset.shape)
dataset = dataset.iloc[:,4:]

# print(dataset.shape)      #(52560, 5)
dataset = np.array(dataset)      #<class 'pandas.core.frame.DataFrame'>
# dataset = dataset.valuse

def split_xy(dataset, x_row, y_row):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + x_row
        y_end_number = x_end_number + y_row

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy(dataset, 336, 48)

# print(x)
# print(y)
print(x.shape)      #(52177, 336, 5)
print(y.shape)      #(52177, 48, 5)
y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])

# 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D, LSTM, GRU, LeakyReLU

model = Sequential()
model.add(Conv1D(filters = 16, kernel_size = 7, strides=1, padding = 'same', input_shape = (336,5), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dense(1))



#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x, y, epochs=1, batch_size=48, verbose=1)

#4. 평가, 예측
result = model.evaluate(x, y, batch_size=48)
print('mse: ', result[0])
print('mae: ', result[1])

# y_pred = model.predict(x_pred)
