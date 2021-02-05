# kfold중 제일 괜찮에 나온 값 hdf5 저장해서 쓰기

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

# ===========================================

skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True)
# Stratified : 타겟값이 같은 것끼리 세트로 만들어 준다. kfold는 이런 속성을 무시하고 접는다.

redu_lr = ReduceLROnPlateau(patience= 80, verbose=1, factor=0.5)
stop = EarlyStopping(monitor='val_accuracy', patience=160, verbose=1, mode='max')

val_loss_min = []
result = 0 
nth = 0

# for train_index, valid_index in skf.split(train2, train['digit']) :
#     x_train = train2[train_index]
#     x_val = train2[valid_index]
#     y_train = train['digit'][train_index]
#     y_val = train['digit'][valid_index]

#     train_generator = idg.flow(x_train, y_train, batch_size=8)
#     valid_generator = idg2.flow(x_val, y_val)

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
#     model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002, epsilon=None), metrics=['accuracy'])
    
#     nth = nth + 1

#     mc = ModelCheckpoint(filepath= '../data/modelcheckpoint/dacon3/3th_best_'+str(nth)+'_{val_accuracy:.4f}.hdf5', save_best_only=True, verbose=1)
#     fit_hist = model.fit_generator(train_generator, epochs=2000, validation_data=valid_generator, callbacks=[stop, redu_lr, mc])

#     print(nth, '번쨰 학습 완료')

# ===========================================
# 예측값 저장
model = load_model('../data/modelcheckpoint/dacon3/3th_best_14_1.0000.hdf5')
# model.summary()

result = model.predict(test2)

# 제출용 저장
sub['digit'] = result.argmax(1)
print(sub.head())

sub.to_csv('../data/csv/dacon3/3th_acc1_3.csv', index = False)
print('======save complete=====')