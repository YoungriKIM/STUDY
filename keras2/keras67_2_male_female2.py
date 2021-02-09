# 실습 / 남자여자 구별
# ImageDataGenerator의 fit 사용해서 완성 / fit으로  하려면 npy로 저장해야함

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten
import matplotlib.pyplot as plt

# npy로 불러오자 -----------------------------------------------------------------------------------------------------

# 적용하기 전에 x_train, y_train, x_test, y_test 저장해 둔 npy를 불러오자
x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_train_y.npy')
x_test = np.load('../data/image/gender/npy/keras67_test_x.npy')
y_test = np.load('../data/image/gender/npy/keras67_test_y.npy')
print('===== load complete =====')


# 훈련을 시켜보자! 모델구성 -----------------------------------------------------------------------------------------------------
model = Sequential()
model.add(Conv2D(32, (7,7), input_shape=(56,56,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
stop = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')

model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2)

#  ----------------------------------------------------------------------------------------------
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

# loss :  19.046550750732422
# acc :  0.675000011920929