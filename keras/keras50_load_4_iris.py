# 저장한 npy 파일을 불러와보자

import numpy as np

x = np.load('../data/npy/iris_x_data.npy')
y = np.load('../data/npy/iris_y_data.npy')

print(x)
print(y)
print(x.shape, y.shape)       # 잘 불러와졌음

# 실습. 모델을 완성하시오
# 아래는 이전 파일에서 불러와서 붙였음

y = np.reshape(y, (150,1))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()

print(y[:5])
print(y.shape)

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

print(x.shape) #(150,4)
print(y.shape) #(150,3)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(120, activation='relu', input_shape=(4,)))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=earlystopping)

loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])
print('y_predict: ', y_predict)
print('y_test[-5:-1]: ', y_test[-5:-1])



#======================= 22-1-1
# loss:  [0.12436151504516602, 0.9666666388511658, 0.04672175273299217]
#======================= 22-1-2
# loss:  [0.11083003133535385, 0.9666666388511658, 0.05501154810190201]

