# keras67_1_male_female1 npy 저장용 파일

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
    ,batch_size=200
    ,class_mode='binary'
    ,subset='training'      
)     

# train_generator
xy_test = train_datagen.flow_from_directory(
    '../data/image/gender' 
    ,target_size=(56,56)
    ,batch_size=200
    ,class_mode='binary'
    ,subset='validation'
)

# npy로 저장하자 -----------------------------------------------------------------------------------------------------
np.save('../data/image/gender/npy/keras67_train_x.npy', arr = xy_train[0][0])
np.save('../data/image/gender/npy/keras67_train_y.npy', arr = xy_train[0][1])
np.save('../data/image/gender/npy/keras67_test_x.npy', arr = xy_test[0][0])
np.save('../data/image/gender/npy/keras67_test_y.npy', arr = xy_test[0][1])
print('===== save complete =====')
