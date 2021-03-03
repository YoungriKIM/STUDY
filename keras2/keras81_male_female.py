# k67의 여성 남성 구분 VGG16으로 바꾸기

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, Xception, InceptionV3
from tensorflow.keras.applications import EfficientNetB0, DenseNet121
from tensorflow.keras.layers import Dense, Flatten, UpSampling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# npy로 불러오고 전처리 -----------------------------------------------------------------------------------------------------

# 적용하기 전에 x_train, y_train, x_test, y_test 저장해 둔 npy를 불러오자
x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_train_y.npy')
x_test = np.load('../data/image/gender/npy/keras67_test_x.npy')
y_test = np.load('../data/image/gender/npy/keras67_test_y.npy')
print('===== load complete =====')

# RGB -> BGR
# from tensorflow.keras.applications.vgg16 import preprocess_input
# x_train = preprocess_input(x_train)
# x_test = preprocess_input(x_test)


# 훈련을 시켜보자! 모델구성 -----------------------------------------------------------------------------------------------------

premodel = DenseNet121(weights='imagenet', include_top = False, input_shape=(224,224,3))
premodel.trainable = False

model = Sequential()
model.add(UpSampling2D(size=(4,4)))
model.add(premodel)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation= 'sigmoid'))

# 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=50, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, mode='max', verbose=1)
filepath = ('../data/modelcheckpoint/k81.hdf5')
mc = ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[stop,lr,mc])

#  ----------------------------------------------------------------------------------------------
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)


# ===========================
# k67 디폴트
# loss :  1.562468409538269 7,7
# acc :  0.6399999856948853

# k81 vgg16
# loss :  1.13721764087677
# acc :  0.7350000143051147
# Xception
# loss :  1.6114599704742432
# acc :  0.8600000143051147
# InceptionV3
# loss :  0.5786319971084595
# acc :  0.875
# EfficientNetB0
# loss :  0.6916043758392334
# acc :  0.5350000262260437
# DenseNet121
# loss :  0.6424410939216614
# acc :  0.8600000143051147

'''
# 따로 저장한 사진으로 예측하기 --------------------------------
filepath = ('../data/modelcheckpoint/k81.hdf5')
model = load_model(filepath)

# 퀸연아
img_pre = load_img('../Users/Admin/Desktop/비트 숙제/0210/yeona.jpg', target_size=(224,224))
pix = img_pre.resize((56,56))
pix = np.array(pix)
img_test = pix.reshape(1, 56, 56, 3)/255.

# 마동석
img_pre_2 = load_img('../Users/Admin/Desktop/비트 숙제/0210/madong.jpg', target_size=(224,224))
pix2 = img_pre_2.resize((56,56))
pix2 = np.array(pix2)
img_test_2 = pix.reshape(1, 56, 56, 3)/255.

# ---------------------------
my_pred_answer = [0]        # 여자
my_pred_no_answer = [1]     # 남자
# ---------------------------

# 퀸연아
my_pred = model.predict(img_test)
print('김연아는(두구두구)')
print((1-my_pred[0][0])*100,'%의 확률로 여자입니다.')

# 마동석
my_pred2 = model.predict(img_test_2)
print('마동석은(두구두구)')
print((my_pred2[0][0])*100,'%의 확률로 남자입니다.')
'''