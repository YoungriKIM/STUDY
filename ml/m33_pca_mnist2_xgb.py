# m31로 만든 0.95 n_component를 사용하여
# dnn 모델을 만들 것
# 기존의 mnist dnn 파일보더 성능을 좋게 만들어라~ + cnn과 비교도 해


import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

#1. x 데이터 불러오고 pca적용 ===================================
(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
# print(x.shape)      #(70000, 28, 28) / x_train = 60000,28,28 / x_test = 10000,28,28
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# print(x.shape)      #(70000, 784)

# pca 적용
pca = PCA(n_components = 713)
x2 = pca.fit_transform(x)
# print(x2.shape)            # (70000, 713)
# pca_EVR = pca.explained_variance_ratio_ # 변화율
# print('pca_EVR: ', pca_EVR)
# print('cumsum: ', sum(pca_EVR))    # cumsum: 0.9500035195029432

# train_test_split
x_train = x2[:60000, :]
x_test = x2[60000:, :]

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train.shape)
# print(x_test.shape)
# (60000, 713)
# (10000, 713)

#1. y 데이터 불러오고 전처리  ===================================
(_, y_train), (_, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)      #(60000, 10) (10000, 10)


#2. 모델 구성 ===================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(200, input_shape=(713, ), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(160, activation='relu'))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='acc', patience=16, mode='max')

model.fit(x_train, y_train, epochs=100, batch_size=28, validation_split=0.2, verbose=1, callbacks=[stop])

#. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=28)
print('loss: ', loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))


# ==============================================================================
# 40-2 mnist CNN
# loss, acc:  0.009633197449147701 0.9853999743461609     137

# 40-3 mnist DNN       
# loss:  [0.10886456072330475, 0.9815000295639038]
# y_pred:  [7 2 1 0 4 1 4 9 6 9]
# y_test:  [7 2 1 0 4 1 4 9 5 9]

# m32_1 pca 0.95이상으로 지정한 파일
# loss:  [0.2978833317756653, 0.9678000211715698]

# m32_2 pca 1.0 이상으로 지정한 파일
# loss:  [0.2978833317756653, 0.9678000211715698]
