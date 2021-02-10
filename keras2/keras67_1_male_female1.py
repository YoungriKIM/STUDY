# 실습 / 남자여자 구별
# ImageDataGenerator의 fit_generator 사용해서 완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------
# 이미지 제너레이터 선언
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=(-0.1,1),
    height_shift_range=(-0.1,1),
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# -----------------------------------------------------------------------------------------------------
# 폴더(디렉토리)에서 불러와서 적용하기! fit과 같다. 이 줄을 지나면 x와 y가 생성이 된다.
# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/gender' 
    ,target_size=(56,56)
    ,batch_size=45
    ,class_mode='binary'
    ,subset='training'      
)     

# train_generator
xy_test = train_datagen.flow_from_directory(
    '../data/image/gender' 
    ,target_size=(56,56)
    ,batch_size=40
    ,class_mode='binary'
    ,subset='validation'
)

# -----------------------------------------------------------------------------------------------------
# print(xy_train[0][0].shape)     #(5, 150, 150, 3)
# print(xy_train[0][1].shape)     #(5,)
# print(xy_train[0][1])           #[1. 0. 1. 0. 1.]
# print(xy_test[0][0].shape)      #(5, 150, 150, 3)
# print(xy_test[0][1].shape)      #(5,)


# 훈련을 시켜보자! 모델구성 -----------------------------------------------------------------------------------------------------
model = Sequential()
model.add(Conv2D(32, (7,7), input_shape=(56,56,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, strides=(2,2)))

model.add(Conv2D(256, (3,3),strides=(1,1), activation='relu',padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(256, (3,3),strides=(1,1), activation='relu',padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, strides=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
stop = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')


history = model.fit_generator(xy_train, steps_per_epoch=4, epochs=500\
                   , validation_data=xy_test, validation_steps=4, callbacks =[stop, lr], verbose=1)

#  ----------------------------------------------------------------------------------------------
loss, acc = model.evaluate(xy_test)
print("loss : ", loss)
print("acc : ", acc)

# 7,7
# loss :  0.5769726037979126
# acc :  0.7060518860816956

