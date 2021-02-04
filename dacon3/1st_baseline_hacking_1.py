# https://dacon.io/competitions/official/235626/codeshare/1669
# 1등 모델 이해해서 재활용하기

# 1st가 한 것 처럼 resizing(64,64) + 3th 모델

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, \
     BatchNormalization, Dropout, experimental
from tensorflow.keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import tensorflow as tf

# dataset 불러오기 ==================================================
train = pd.read_csv('../data/csv/dacon3/train.csv')
test = pd.read_csv('../data/csv/dacon3/test.csv')
sub = pd.read_csv('../data/csv/dacon3/submission.csv')

# 필요없는 부분 떨구기
train2 = train.drop(['id', 'digit', 'letter'], axis = 1).values
test2 = test.drop(['id', 'letter'], axis = 1).values

# print(train2.shape)     #(2048, 784)
# print(test2.shape)      #(20480, 784)

# reshape + scaler
train2 = train2.reshape(-1, 28, 28, 1)/255
test2 = test2.reshape(-1, 28, 28, 1)/255

# print(train2.shape)     #(2048, 28, 28, 1)
# print(test2.shape)      #(20480, 28, 28, 1)

# 리사이징
train2 = experimental.preprocessing.Resizing(64,64)(train2)
test2 = experimental.preprocessing.Resizing(64,64)(test2)

# print(type(train2)) #<class 'tensorflow.python.framework.ops.EagerTensor'>
# 타입이 안 맞으면 for문에 들어가지 않아서 numpy.ndarray로 바꿔줘야 함

train2 =  np.array(train2)
test2 =  np.array(test2)

# print(type(train2)) #<class 'numpy.ndarray'>
# 해결

# print(train2.shape)     #(2048, 256, 256, 1)
# print(test2.shape)      #(20480, 256, 256, 1)

'''
# 그래프로 잘 커졌나 확인 ------------------------------------------------
plt.figure(figsize = (12, 4), dpi = 80)

plt.subplot(2,2,1)
plt.imshow(tf.squeeze(train2[311]), cmap = 'jet')
plt.title(f'{tf.squeeze(train2[311]).shape}')

plt.subplot(2,2,2)
plt.imshow(tf.squeeze(resizing[311]), cmap = 'jet')
plt.title(f'{tf.squeeze(resizing[311]).shape}')

plt.tight_layout()
plt.show()
# ---------------------------------------------------------------------
'''

# 이미지 증폭 정의 / idg2는 증폭없이 형태만 맞춰줌 
idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1, 1))
idg2 = ImageDataGenerator()

# kfold 정의
skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True)

# callback 정의
redu_lr = ReduceLROnPlateau(patience= 80, verbose=1, factor=0.5)
stop = EarlyStopping(monitor='val_loss', patience=160, verbose=1, mode='min')
mc = ModelCheckpoint(filepath= '../data/modelcheckpoint/dacon3/1st_01.h5', save_best_only=True, verbose=1)

result = 0 
nth = 0


# for문으로 모델 + 컴파일 + 훈련 + 평가

for train_index, valid_index in skf.split(train2, train['digit']) :
    x_train = train2[train_index]
    x_val = train2[valid_index]
    y_train = train['digit'][train_index]
    y_val = train['digit'][valid_index]
    
    # print(x_train.shape)       #(1997, 256, 256, 1)
    # print(x_val.shape)         #(51, 256, 256, 1)
    # print(y_train.shape)       #(1997,)
    # print(y_val.shape)         #(51,)

    train_generator = idg.flow(x_train, y_train, batch_size=8)
    valid_generator = idg2.flow(x_val, y_val)
    test_generator = idg2.flow(test2,shuffle=False)

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

    # 컴파일, 훈련
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002, epsilon=None), metrics=['acc'])
    #   알아서 원핫인코딩 해주는 기능 / sparse_categorical_crossentropy
    fit_hist = model.fit_generator(train_generator, epochs=2000, validation_data=valid_generator, callbacks=[stop, redu_lr, mc])

    # predict
    model.load_weights('../data/modelcheckpoint/dacon3/1st_01.h5')
    result += model.predict_generator(test_generator,verbose=True)/40

    nth += 1
    print(nth, '번쨰 학습 완료')

model.summary()

# =====================================================

# # 예측
# model = load_model('../data/modelcheckpoint/dacon3/1st_01.h5')
# model.summary()

# result = model.predict(test2)

# 제출용 저장
sub['digit'] = result.argmax(1)
print(sub.head())

sub.to_csv('../data/csv/dacon3/1st_01.csv', index = False)
print('======save complete=====')

#=====================================================
# epochs = 2000 > 3th_0202_2.csv > dacon score : 0.1078431373  이런 trash
# 03 수정중 > test 부분 imagegenerate 안하고 (n, 28, 28 ,1)로 넣음 > 3th_0202_3 > daconscore : 0.9264705882	
# 1st가 한 것 처럼 resizing + 3th 모델 = 1st_01.csv >  1st_01 > daconscore : 0.9264705882 > 리사이징 큰 효과 없음 