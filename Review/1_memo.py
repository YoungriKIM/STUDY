# 나를 찍어서 내가 남자인지 여자인지에 대해
# 나를 찍어서 여자일 확률이 몇프로 인지 출력
# 선생님에게 메일 보낼때 로직이랑 정확도 보내기


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
# ImageDataGenerator은 randomsearch보다 빠르고 파라미터도 자동으로 부여해준다 

data_generator = ImageDataGenerator(
    rescale = 1. / 255, 
    shear_range = 0.2, 
    zoom_range = 0.2, 
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 180,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    validation_split = 0.2) 

train_generator = data_generator.flow_from_directory(
    'C:/data/image/ma_female/classifer', 
    target_size =(64, 64), 
    batch_size = 16,
    shuffle = True,
    class_mode = 'categorical',
    seed = 42,
    subset='training')

validation_generator = data_generator.flow_from_directory( 
    'C:/data/image/ma_female/classifer', 
    target_size =(64, 64), 
    batch_size = 16,
    shuffle = True,
    class_mode = 'categorical',
    seed = 42,
    subset='validation')

X_train, y_train = next(train_generator)
X_test, y_test = next(validation_generator)

#============== ImageDataGenerator정의 ====================
# train ---> 학습필수
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    validation_split= 0.3
) 

# test --> 전처리만
test_datagen = ImageDataGenerator(rescale=1./255)

# ===================== generator ===========================
# train

xy_train = train_datagen.flow_from_directory(
    'C:/data/image/ma_female/classifer',
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

# validation
xy_val = train_datagen.flow_from_directory(
    'C:/data/image/ma_female/classifer',
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

print(xy_train[0][0].shape) # (16, 64, 64, 3)
print(xy_train[0][1].shape) # (16,)

# ======================== model ===========================
from keras.optimizers import Adam
model = Sequential([
    # 1st conv
  Conv2D(96, (3,3),strides=(4,4), padding='same', activation='relu', input_shape=(64, 64, 3)),
  BatchNormalization(),
  MaxPooling2D(2, strides=(2,2)),
    # 2nd conv
  Conv2D(256, (3,3),strides=(1,1), activation='relu',padding="same"),
  BatchNormalization(),
    # 3rd conv
  Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  BatchNormalization(),
  MaxPooling2D(2, strides=(2, 2)),
    # To Flatten layer
  Flatten(),
  Dropout(0.5),
    #To FC layer 1
  Dense(30, activation='relu'),
  Dropout(0.5),
  Dense(10, activation='relu'),
  Dense(1, activation='sigmoid')
  ])

# ===================== 훈련 ==============================
model.compile(
    optimizer=Adam(lr=0.001),
    loss='binary_crossentropy',
    metrics=['acc']
   )
modelpath = '../data/modelCheckpoint/k67_4_{epoch:02d}-{val_loss:.4f}.hdf5'
es= EarlyStopping(monitor='val_loss', patience=5)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                    save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, mode='auto')
history = model.fit_generator(generator=xy_train,
                    validation_data=xy_val,
                    epochs=150,
                    callbacks=[es, cp, lr])
acc = history.history['acc']
val_acc = history.history['val_acc']
print("val_acc :", np.mean(val_acc))
print('acc: ', acc[-1])

model.save('../data/h5/classifer.h5')


# ==================== 남녀 구분 이미지로 나타내기 ==========================
# https://thecleverprogrammer.com/2020/11/25/gender-classification-with-python/
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

# predicting images
path = "C:/data/image/ma_female/classifer/female/final_1000.jpg"
img = image.load_img(path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=1)
print(classes[0])
if classes[0]>0.5:
    print("그는 남자다")
else:
    print( "그녀는 여자다")
plt.imshow(img)
plt.show()