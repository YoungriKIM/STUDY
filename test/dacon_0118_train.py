import numpy as np
import pandas as pd

dataset = pd.read_csv('../data/csv/dacon1/train/train.csv', index_col=None, header=0)
# print(dataset.shape)
dataset = dataset.iloc[:,3:]

# print(dataset.shape)      #(52560, 5)
dataset = np.array(dataset)      #<class 'pandas.core.frame.DataFrame'>

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

x, y = split_xy(dataset, 336, 96)

# print(x)
# print(y)
print(x.shape)      #(52129, 336, 5)
print(y.shape)      #(52129, 96, 5)

x = x.reshape(x.shape[0], 7, int(x.shape[1]/7), x.shape[2])
y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])

print(x.shape)      #(52129, 7, 48, 5)
print(y.shape)      #(52129, 480)


# 테스트 불러오기
pred_data = pd.read_csv('../data/csv/dacon1/test/0.csv', index_col=None, header=0)
# print(dataset.shape)
x_pred = pred_data.iloc[:,3:].values
print(x_pred.shape)      #(336, 5)

x_pred = x_pred.reshape(1, 7, int(x_pred.shape[0]/7), x_pred.shape[1])
print(x_pred.shape)      #(1, 7, 48, 5)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=1, padding='same', input_shape=(x.shape[1], x.shape[2], x.shape[3]), activation='relu'))
model.add(MaxPooling2D(pool_size=4))
model.add(Flatten())
model.add(Dense(576))

#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x, y, epochs=20, batch_size=48, verbose=1)

#4. 평가, 예측
result = model.evaluate(x, y, batch_size=48)
print('mse: ', result[0])
print('mae: ', result[1])

y_pred = model.predict(x_pred)
# print(y_pred)
y_pred = y_pred.reshape(96,6)

df = pd.DataFrame(y_pred)
# df.to_csv('../data/csv/dacon0.csv', sep=',') 

# mse:  9698.716796875
# mae:  36.95664596557617