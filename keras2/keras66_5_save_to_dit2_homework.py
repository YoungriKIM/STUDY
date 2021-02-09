## k66_5 flow_from_directory 말고 flow로 dir 저장해라

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL.Image as pilimg


# imagedatagenerator 선언 --------------------------------------
# train 용
train_datagen = ImageDataGenerator()
# test 용
test_datagen = ImageDataGenerator()
# 불러올 npy 파일에 imagegenerator가 적용되어 있어 형식만 맞춤

# 적용하기 전에 x_train, y_train, x_test, y_test 저장해 둔 npy를 불러오자
x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')
print('===== load complete =====')

# imagedatagenerator 적용 / flow_from_directory 말고 flow로 하자 --------------------------------------
# xy_train > flow
xy_train = train_datagen.flow(x_train, y_train,
                              batch_size=5,
                              save_to_dir = '../data/image/brain_generator/train')

print('===== save complete =====')

# xy_test > flow
xy_test = test_datagen.flow(x_test, y_test, batch_size=5)

# 잘 생성이 됐는지 찍어보자 ------------------------------------------------------------------------
print(xy_train[0][0].shape)
# (5, 150, 150, 3)
print(xy_train[0][1].shape)
# (5,)

# 저장된 파일 확인!