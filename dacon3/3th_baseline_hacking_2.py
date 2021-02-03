# https://dacon.io/competitions/open/235626/codeshare/1682
# 이해해서 재활용하기

# 1에서 수정 / 설명 첨부하고 / 스코어 나오게 수정

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
train = pd.read_csv('../data/csv/dacon3/train.csv')
test = pd.read_csv('../data/csv/dacon3/test.csv')
sub = pd.read_csv('../data/csv/dacon3/submission.csv')

# 필요없는 부분 떨구기
train2 = train.drop(['id', 'digit', 'letter'], axis = 1).values
test2 = test.drop(['id', 'letter'], axis = 1).values

# plt.imshow(train2[100].reshape(28,28))
# plt.show()

# 쉐잎 맞추고 , 민맥스 스케일러
train2 = train2.reshape(-1, 28, 28, 1)/255
test2 = test2.reshape(-1, 28, 28, 1)/255

# print(train2.shape)     #(2048, 28, 28, 1)
# print(test2.shape)      #(20480, 28, 28, 1)

idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

# -----------------------------------
# # ImageDataGenerator 미리보기

# # 복사해서 한 장 가져오기
# sample_data = train2[100].copy()
# sample = expand_dims(sample_data, axis = 0) # expand_dims : Expand the shape of an array.

# # print(train2[100].shape)    #(28, 28, 1)
# # print(sample.shape)         #(1, 28, 28, 1)

# sample_datagen = ImageDataGenerator(height_shift_range=(-1,1), width_shift_range=(-1,1))
# sample_generator = sample_datagen.flow(sample, batch_size=1)

# plt.figure(figsize=(16,10)) # 도화지

# for i in range(9) :
#     plt.subplot(3, 3, i+1)
#     sample_batch = sample_generator.next()
#     sample_image = sample_batch[0]
#     plt.imshow(sample_image.reshape(28,28))

# # plt.show()
# -----------------------------------

skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True)
# Stratified : 타겟값이 같은 것끼리 세트로 만들어 준다. kfold는 이런 속성을 무시하고 접는다.

redu_lr = ReduceLROnPlateau(patience= 80, verbose=1, factor=0.5)
stop = EarlyStopping(monitor='val_loss', patience=160, verbose=1, mode='min')
mc = ModelCheckpoint(filepath= '../data/modelcheckpoint/dacon3/3th_02.h5', save_best_only=True, verbose=1)

val_loss_min = []
result = 0 
nth = 0

# for train_index, valid_index in skf.split(train2, train['digit']) :
#     x_train = train2[train_index]
#     x_val = train2[valid_index]
#     y_train = train['digit'][train_index]
#     y_val = train['digit'][valid_index]
    
#     print(x_train.shape)       #(1997, 28, 28, 1)
#     print(x_val.shape)         #(51, 28, 28, 1)
#     print(y_train.shape)       #(1997,)
#     print(y_val.shape)         #(51,)

#     train_generator = idg.flow(x_train, y_train, batch_size=8)
#     valid_generator = idg2.flow(x_val, y_val)

#     print(train_generator.shape) 
#     print(valid_generator.shape) 
#     # 여기까지 보다가 위로 올라감

#     model = Sequential()
#     model.add(Conv2D(16, (3,3), input_shape=(x_train.shape[1:]), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.2))
    
#     model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
#     model.add(BatchNormalization())
#     model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((3,3)))
#     model.add(Dropout(0.3))
    
#     model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((3,3)))
#     model.add(Dropout(0.3))
    
#     model.add(Flatten())

#     model.add(Dense(128,activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.3))
#     model.add(Dense(64,activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.3))

#     model.add(Dense(10,activation='softmax'))

#     # 컴파일, 훈련
#     model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002, epsilon=None), metrics=['acc'])
      # 알아서 원핫인코딩 해주는 기능 / sparse_categorical_crossentropy
#     fit_hist = model.fit_generator(train_generator, epochs=2000, validation_data=valid_generator, callbacks=[stop, redu_lr, mc])

#     nth = nth + 1
#     print(nth, '번쨰 학습 완료')

# model.summary()

# =====================================================

# 예측
model = load_model('../data/modelcheckpoint/dacon3/3th_02.h5')
model.summary()

result = model.predict(test2)

# 제출용 저장
sub['digit'] = result.argmax(1)
print(sub.head())

sub.to_csv('../data/csv/dacon3/3th_0202_3.csv', index = False)
print('======save complete=====')

#=====================================================
# epochs = 2000 > 3th_0202_2.csv > dacon score : 0.1078431373  이런 trash
# 03 수정중 > test 부분 imagegenerate 안하고 (n, 28, 28 ,1)로 넣음 > 3th_0202_3 > daconscore : 0.9264705882	

