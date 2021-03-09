import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten
import pandas_profiling

df = pd.read_csv('../data/csv/winequality-white.csv', sep=';')
print(df.head())

x_data = df.iloc[:,[0,1,2,3,4,5,6,7,9]]
y_data = df.iloc[:,-1]

x = x_data.values
y = np.array([0 if i<=5 else 1 if i==6 else 2 for i in y_data])

# 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=311)

# 범위 0~1사이로
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, \
                                  QuantileTransformer, MaxAbsScaler, PowerTransformer
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 벡터화하고
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
y_train =y_train.reshape(-1,1)
y_test =y_test.reshape(-1,1)
y_val =y_val.reshape(-1,1)

hot = OneHotEncoder()
hot.fit(y_train)
y_train = hot.transform(y_train).toarray()
y_test = hot.transform(y_test).toarray()
y_val = hot.transform(y_val).toarray()

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)
# (3134, 11)
# (980, 11)
# (784, 11)
# (3134, 3)
# (980, 3)
# (784, 3)

# --------------------------------------------------
# Conv1D 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

model = Sequential()
model.add(Conv1D(filters=48, kernel_size=2, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(36, 1))
model.add(Conv1D(36, 1))
model.add(Flatten())
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
earlystopping = EarlyStopping(monitor='val_loss', patience=12, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=6, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=200, batch_size=4, validation_data=(x_val, y_val), verbose=2, callbacks=[earlystopping, lr])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=4)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])
print('y_predict_argmax: ', y_predict.argmax(axis=1)) 
print('y_test[-5:-1]_argmax: ', y_test[-5:-1].argmax(axis=1)) 


# =================================
# QuantileTransformer # val_data 나눔, lr추가
# loss:  [1.031842827796936, 0.563265323638916]
# loss:  [0.8284953832626343, 0.6000000238418579]

# loss:  [0.23526695370674133, 0.9285714030265808]