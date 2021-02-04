# adam 불러와서 디폴트 크게 잡고 시작
# + 튜닝 조금 / 배치랑 드랍아웃 같이 안쓰게 / 이미지증폭 8 >10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 데이터 맨 처음 지정 ===================
train = pd.read_csv('../data/csv/dacon3/train.csv')
test = pd.read_csv('../data/csv/dacon3/test.csv')
sub = pd.read_csv('../data/csv/dacon3/submission.csv')

x_origin = train.drop(['id', 'digit', 'letter'], axis=1).values
all_test = test.drop(['id', 'letter'], axis=1).values

#------- for conv2d
x_origin = x_origin.reshape(-1, 28, 28, 1)/255
all_test = all_test.reshape(-1, 28, 28, 1)/255

# 이미지 증폭 정의
idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

# kfold 정의
skf = StratifiedKFold(n_splits=32, random_state=311, shuffle=True)

# ====================================================================
# 모델 구성 --- Conv2D ---

val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(x_origin, train['digit']) :
    x_train = x_origin[train_index]
    x_val = x_origin[valid_index]
    y_train = train['digit'][train_index]
    y_val = train['digit'][valid_index]

    train_generator = idg.flow(x_train, y_train, batch_size=10)
    valid_generator = idg2.flow(x_val, y_val)
    test_generator = idg2.flow(all_test, shuffle=False)

    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=2, strides=1, padding='same', \
        input_shape=(x_train.shape[1:]), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(64, 2, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 2, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(32, 2, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32,activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(10,activation='softmax'))

    # ====================================================================

    #3. 컴파일, 훈련
    model.compile(loss='sparse_categorical_crossentropy',  optimizer=Adam(lr=0.002,epsilon=None), metrics=['acc'])

    redu_lr = ReduceLROnPlateau(patience= 64, verbose=1, factor=0.5)
    stop = EarlyStopping(monitor='val_acc', patience=128, verbose=1, mode='max')
    mc = ModelCheckpoint(filepath= '../data/modelcheckpoint/dacon3/self_0204_3.h5', save_best_only=True, verbose=1)

    history = model.fit_generator(train_generator, epochs=2000, validation_data=(valid_generator), verbose=1, callbacks=[stop, redu_lr, mc])

    # 예측
    model.load_weights('../data/modelcheckpoint/dacon3/self_0204_3.h5')
    result += model.predict_generator(test_generator, verbose=True)/40
    
    nth += 1
    print(nth, '번째 학습')

# ====================================================================

# 제출용 저장

sub['digit'] = result.argmax(1)
print(sub.head())

sub.to_csv('../data/csv/dacon3/self_0204_3.csv', index = False)
print('======save complete=====')

# =============================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
# =============================================
# self_0204_2 > dacon score : 0.9117647059	
# self_0204_3 > ing