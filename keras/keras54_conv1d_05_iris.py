# Conv1D로 바꾸시오 그리고 비교하시오

import numpy as np
from sklearn.datasets import load_iris 

dataset = load_iris()
x = dataset.data
y = dataset.target
y = y.reshape(y.shape[0], 1)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()

# print(y[:5])
# print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# print(x_train.shape)    #(96, 4)
# print(x_val.shape)      #(24, 4)
# print(x_test.shape)     #(30, 4)
# print(y_train.shape)    #(96, 3)
# print(y_val.shape)      #(24, 3)
# print(y_test.shape)     #(30, 3)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D

model = Sequential()
model.add(Conv1D(filters=48, kernel_size=2, input_shape=(4, 1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(36, 1))
model.add(Conv1D(36, 1))
model.add(Flatten())
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=earlystopping)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)

# y_predict = model.predict(x_test[-5:-1])
# print('y_predict: ', y_predict.argmax(axis=1))
# print('y_test[-5:-1]: ', y_test[-5:-1].argmax(axis=1))



#======================= 22-1-1
# loss:  [0.12436151504516602, 0.9666666388511658, 0.04672175273299217]
#======================= 22-1-2
# loss:  [0.11083003133535385, 0.9666666388511658, 0.05501154810190201]

# 38-4 드랍아웃 적용 (loss랑 mae는 조금 줄어들었음, acc는 동일함)
# loss:  [0.1093134954571724, 0.9666666388511658, 0.03586991876363754]

# 41-4 iris CNN         더 좋아짐
# loss:  [0.0934637114405632, 0.9666666388511658, 0.041757065802812576]

# 54-5 conv1d
# loss:  [0.3125566840171814, 0.8999999761581421, 0.11018746346235275]