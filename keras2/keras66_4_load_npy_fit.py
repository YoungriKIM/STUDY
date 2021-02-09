# 이미지데이터를 불러와서 증폭을 해보자! > 적용한 걸 훈련시키자 > npy로 저장하자 > 저장한 걸 불러와서 핏하자

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# npy로 불러오자 -----------------------------------------------------------------------------------------------------
x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')
print('===== load complete =====')

print(x_train.shape, y_train.shape)
# (160, 150, 150, 3) (160,)
print(x_test.shape, y_test.shape)
# (120, 150, 150, 3) (120,)

# 실습, 모델을 만들어라

# 모델
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(150,150,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

# 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

# 평가
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss, acc: ', loss, acc)

# =========================
# loss, acc:  14.118191719055176 0.5083333253860474