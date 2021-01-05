import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터 제공
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target

# print(x.shape, y.shape) #(150, 4) (150,)
# print(x[:5])
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]
# print(y) #0, 1, 2가 있어 분류 종류가 3가지이다.

# 전처리 : Y onehotencording or to_categorical / X train_test_split / X MinMaxScaler

#y전처리 텐서플로용
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

#y전처리 사이킷런용
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# ohe.fit(y)
# y = ohe.transform(y).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, suffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, suffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(40, input_shape=(4,), activation='relu'))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='biary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(moitor='loss', patience=20, mode='min')
model.fi(x_train, y_train, epochs=1000, batch_size=8, verbose=2, validation_data=(x_val, y_val), callbacks=[stop])
