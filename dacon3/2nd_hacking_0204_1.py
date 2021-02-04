# 2등 모델 해킹
# 참고 : https://dacon.io/competitions/official/235626/codeshare/1679?page=1&dtype=recent&ptype=pub

# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from keras.optimizers import Adam, RMSprop
from tensorflow.keras import optimizers
from keras.utils import np_utils
from keras import backend as bek
import cv2 #openCV
import gc
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics


# dataset 불러오기
train = pd.read_csv('../data/csv/dacon3/train.csv')
test = pd.read_csv('../data/csv/dacon3/test.csv')

print('가능한~')

# 필요 없는 칼럼 버리고 +넘파이화 +리쉐잎
x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28, 1)
# 0~20 사이 0으로 수렴
x_train = np.where((x_train<=20)&(x_train!=0) ,0.,x_train)
# scaler + float32로 변경
x_train = x_train/255
x_train = x_train.astype('float32')

# y 지정
y = train['digit']
# y 벡터화 0~9까지 10가지
y_train = np.zeros((len(y), len(y.unique())))  # 총 행의수 , 10(0~9)
for i, digit in enumerate(y):
    y_train[i, digit] = 1

# x_train 컬러화 + 특성 증폭 ------------
# 까만 도화지 준비
train_224=np.zeros([2048,56,56,3],dtype=np.float32)

for i, s in enumerate(x_train):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(converted,(56,56),interpolation = cv2.INTER_CUBIC)
    del converted
    train_224[i] = resized
    del resized
    bek.clear_session()
    gc.collect()

print('가능한~2')

# 이미지 제너레이터 정의
datagen = ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.15,
        rotation_range = 10,
        validation_split=0.2)

valgen = ImageDataGenerator()

# callbacks + 모델 정의
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
def create_model() :
    
  effnet = tf.keras.applications.EfficientNetB3(
      include_top=True,
      weights=None,
      input_shape=(300,300,3),
      classes=10,
      classifier_activation="softmax",
  )
  model = Sequential()
  model.add(effnet)


  model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(lr=initial_learningrate),
              metrics=['accuracy'])
  return model

initial_learningrate=2e-3  
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=40)
cvscores = []
Fold = 1
results = np.zeros((20480,10) )
def lr_decay(epoch):#lrv
    return initial_learningrate * 0.99 ** epoch


# 테스트 데이터 지정 ````
x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = np.where((x_test<=20)&(x_test!=0) ,0.,x_test)
# x_test = np.where(x_test>=145,255.,x_test)
x_test = x_test/255
x_test = x_test.astype('float32')

test_224=np.zeros([20480,56,56,3],dtype=np.float32)


for i, s in enumerate(x_test):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(converted,(56,56),interpolation = cv2.INTER_CUBIC)
    del converted
    test_224[i] = resized
    del resized

bek.clear_session()
gc.collect()

# 최종 데이터 정의
results = np.zeros((20480,10),dtype=np.float32)

for train, val in kfold.split(train_224): 
    # if Fold<25:
    #   Fold+=1
    #   continue
    
    initial_learningrate=2e-3  
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=50)      
    filepath_val_acc="/content/MyDrive/My Drive/Colab Notebooks/models/effi_model_aug"+str(Fold)+".ckpt"
    checkpoint_val_acc = ModelCheckpoint(filepath_val_acc, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)

    gc.collect()
    bek.clear_session()
    print ('Fold: ',Fold)
    
    X_train = train_224[train]
    X_val = train_224[val]
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    Y_train = y_train[train]
    Y_val = y_train[val]

    model = create_model()


    training_generator = datagen.flow(X_train, Y_train, batch_size=32,seed=7,shuffle=True)
    validation_generator = valgen.flow(X_val, Y_val, batch_size=32,seed=7,shuffle=True)
    model.fit(training_generator,epochs=150,callbacks=[LearningRateScheduler(lr_decay),es,checkpoint_val_acc],
               shuffle=True,
               validation_data=validation_generator,
               steps_per_epoch =len(X_train)//32
               )
    del X_train
    del X_val
    del Y_train
    del Y_val

    gc.collect()
    bek.clear_session()
    model.load_weights(filepath_val_acc)
    results = results + model.predict(test_224)
    
    Fold = Fold +1
    
submission = pd.read_csv('/content/MyDrive/My Drive/Colab Notebooks/data/submission.csv')
submission['digit'] = np.argmax(results, axis=1)
submission.head()
submission.to_csv('/content/MyDrive/My Drive/Colab Notebooks/loadtest2.csv', index=False)

# ======================================================