# https://dacon.io/competitions/open/235626/codeshare/1682
# 이해해서 재활용하기

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

train = pd.read_csv('../data/csv/dacon3/train.csv')
test = pd.read_csv('../data/csv/dacon3/test.csv')
sub = pd.read_csv('../data/csv/dacon3/submission.csv')

train2 = train.drop(['id', 'digit', 'letter'], axis = 1).values
test2 = test.drop(['id', 'letter'], axis = 1).values

# plt.imshow(train2[100].reshape(28,28))
# plt.show()

train2 = train2.reshape(-1, 28, 28, 1)/255
test2 = test2.reshape(-1, 28, 28, 1)/255

print(train2.shape)
print(test2.shape)

idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

# -----------------------------------
sample_data = train2[100].copy()
sample = expand_dims(sample_data, axis = 0) # expand_dims : Expand the shape of an array.

# print(train2[100].shape)    #(28, 28, 1)
# print(sample.shape)         #(1, 28, 28, 1)

sample_datagen = ImageDataGenerator(height_shift_range=(-1,1), width_shift_range=(-1,1))
sample_generator = sample_datagen.flow(sample, batch_size=1)

plt.figure(figsize=(16,10))

for i in range(9) :
    plt.subplot(3, 3, i+1)
    sample_batch = sample_generator.next()
    sample_image = sample_batch[0]
    plt.imshow(sample_image.reshape(28,28))

# plt.show()
# -----------------------------------

skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True)
# Stratified : 타겟값이 같은 것끼리 세트로 만들어 준다. kfold는 이런 속성을 무시하고 접는다.

redu_lr = ReduceLROnPlateau(patience= 80, verbose=1, factor=0.5)
stop = EarlyStopping(patience=160, verbose=1)
mc = ModelCheckpoint(filepath= '../data/modelcheckpoint/dacon3/3th_02.h5', save_best_only=True, verbose=1)

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
    fit_hist = model.fit_generator(train_generator, epochs=2000, validation_data=valid_generator, callbacks=[stop, redu_lr, mc])

    nth = nth + 1
    print(nth, '번쨰 학습 완료')

model.summary()

# =====================================================

# 예측
# model = load_model('../data/modelcheckpoint/dacon3/3th_02.h5')
# model.summary()

# test_generator = idg2.flow(test2, shuffle=True)

# result = (model.predict_generator(test_generator, verbose= True)/40) + 1

# # 제출용 저장
# sub['digit'] = result.argmax(1)
# print(sub.head())

# sub.to_csv('../data/csv/dacon3/3th_0202_1.csv', index = False)

#=====================================================
# ing > epoch 늘려서 해라..젭알
# 2000으로 돌리는 중