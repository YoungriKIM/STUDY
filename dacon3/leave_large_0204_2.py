# 특성이 큰 것만 남긴 데이터를 쓰겠음 버전2
# 참고 : https://dacon.io/competitions/official/235626/codeshare/1624?page=3&dtype=recent&ptype=pub

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam


# dataset 불러오기
train = pd.read_csv('/content/drive/MyDrive/colab_data/dacon3/train.csv')
test = pd.read_csv('/content/drive/MyDrive/colab_data/dacon3/test.csv')
sub = pd.read_csv('/content/drive/MyDrive/colab_data/dacon3/submission.csv')

# 필요없는 부분 떨구기
train2 = train.drop(['id', 'digit', 'letter'], axis = 1).values
test2 = test.drop(['id', 'letter'], axis = 1).values

# 쉐잎 맞추고 , 민맥스 스케일러
train2 = train2.reshape(-1, 28, 28, 1)/255
test2 = test2.reshape(-1, 28, 28, 1)/255

# 특성 작은값 0으로 수렴
threshold = 0.3
train2[train2 < threshold] = 0
test2[test2 < threshold] = 0

idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

skf = StratifiedKFold(n_splits=40, random_state=311, shuffle=True)
# Stratified : 타겟값이 같은 것끼리 세트로 만들어 준다. kfold는 이런 속성을 무시하고 접는다.

val_loss_min = []
result = 0 
nth = 0

for train_index, valid_index in skf.split(train2, train['digit']) :
    x_train = train2[train_index]
    x_val = train2[valid_index]
    y_train = train['digit'][train_index]
    y_val = train['digit'][valid_index]

    train_generator = idg.flow(x_train, y_train, batch_size=8)
    valid_generator = idg2.flow(x_val, y_val)
    test_generator = idg2.flow(test2, shuffle=False)

    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape=(x_train.shape[1:]), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(10,activation='softmax'))

    # ====================================================================

    #3. 컴파일, 훈련
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

    redu_lr = ReduceLROnPlateau(patience= 64, verbose=1, factor=0.1)
    stop = EarlyStopping(monitor='val_acc', patience=128, verbose=1, mode='max')
    mc = ModelCheckpoint(filepath= '/content/drive/MyDrive/colab_data/modelcheckpoint/dacon3/large_0204_2.h5', save_best_only=True, verbose=1)

    history = model.fit_generator(train_generator, epochs=2000, validation_data=(valid_generator), verbose=1, callbacks=[stop, redu_lr, mc])

    # 예측
    model.load_weights('/content/drive/MyDrive/colab_data/modelcheckpoint/dacon3/large_0204_2.h5')
    result += model.predict_generator(test_generator, verbose=True)/40
    
    nth += 1
    print(nth, '번째 학습')

# ====================================================================

# 제출용 저장

sub['digit'] = result.argmax(1)
print(sub.head())

sub.to_csv('/content/drive/MyDrive/colab_data/dacon3/large_0204_2.csv', index = False)
print('======save complete=====')

# =============================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
# =============================================

#=====================================================
# epochs = 2000 > 3th_0202_2.csv > dacon score : 0.1078431373  이런 trash
# 03 수정중 > test 부분 imagegenerate 안하고 (n, 28, 28 ,1)로 넣음 > 3th_0202_3 > daconscore : 0.9264705882	
# 특성 작은 값 0으로 수렴해서 0.1 >  large_0204_1 > dacon score: 0.93627450
# 특성 작은 값 0으로 수렴해서2  0.3 > large_0204_2 > dacon score: 0.9068627451	
