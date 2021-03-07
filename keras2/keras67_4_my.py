# 나를 찍어서 내가 남자인지 여자인지 또 여자일 acc는 몇인지

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import PIL.Image as pilimg
from sklearn.metrics import accuracy_score

# npy로 불러오자 -----------------------------------------------------------------------------------------------------

# 적용하기 전에 x_train, y_train, x_test, y_test 저장해 둔 npy를 불러오자
x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_train_y.npy')
x_test = np.load('../data/image/gender/npy/keras67_test_x.npy')
y_test = np.load('../data/image/gender/npy/keras67_test_y.npy')
print('===== load complete =====')


# 훈련을 시켜보자! 모델구성 -----------------------------------------------------------------------------------------------------
# model = Sequential()
# model.add(Conv2D(16, (7,7), input_shape=(x_train.shape[1:]), padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

model = load_model('../data/modelcheckpoint/k67_-0.6500.hdf5')

# 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='val_acc', patience=20, mode='auto')
lr = ReduceLROnPlateau(monitor='val_lacc', factor=0.3, patience=10, mode='max')
filepath = ('../data/modelcheckpoint/k67_-{val_acc:.4f}.hdf5')
mc = ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=1)

# model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[stop,lr])#,mc])

#  ----------------------------------------------------------------------------------------------
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)


# ===========================
# loss :  1.562468409538269 7,7
# acc :  0.6399999856948853
# ===========================



# 내 사진 불러와서 사이즈 맞추기
# image = pilimg.open('../Users/Admin/Desktop/비트 숙제/0210/sample_image.jpg')
# pix = image.resize((56,56))
# pix = np.array(pix)
# my_test = pix.reshape(1, 56, 56, 3)/255.
# ---------------------------
# 퀸연아로 맞추기
image2 = pilimg.open('../Users/Admin/Desktop/비트 숙제/0210/yeona.jpg')
pix2 = image2.resize((56,56))
pix2 = np.array(pix2)
yeona_test = pix2.reshape(1, 56, 56, 3)/255.
# ---------------------------
# 레오나르도 디카프리오
image3 = pilimg.open('../Users/Admin/Desktop/비트 숙제/0210/actor.jpg')
pix3 = image3.resize((56,56))
pix3 = np.array(pix3)
actor_test = pix3.reshape(1, 56, 56, 3)/255.
# ---------------------------
# 마동석
image4 = pilimg.open('../Users/Admin/Desktop/비트 숙제/0210/madong.jpg')
pix4 = image4.resize((56,56))
pix4 = np.array(pix4)
madong_test = pix4.reshape(1, 56, 56, 3)/255.


# ---------------------------
my_pred_answer = [0]        # 여자
my_pred_no_answer = [1]     # 남자
# ---------------------------


# 예측하기 _1
# my_pred = model.predict(my_test)
# print('당신은(두구두구)')
# print((1-my_pred[0][0])*100,'%의 확률로 여자입니다.')       #99.99978571840984 %의 확률로 여자입니다.

# 예측하기 _2
my_pred2 = model.predict(yeona_test)
print('김연아는(두구두구)')
print((1-my_pred2[0][0])*100,'%의 확률로 여자입니다.')      #99.99999874083656 %의 확률로 여자입니다.

# 예측하기 _3
my_pred3 = model.predict(actor_test)
print('레오나르도 디카프리오는(두구두구)')
print((my_pred3[0][0])*100,'%의 확률로 남자입니다.')        #0.0069707057264167815 %의 확률로 남자입니다.

# 예측하기 _4
my_pred4 = model.predict(madong_test)
print('마동석은(두구두구)')
print((my_pred4[0][0])*100,'%의 확률로 남자입니다.')        #88.35261464118958 %의 확률로 남자입니다.

