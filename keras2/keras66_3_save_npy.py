# 이미지데이터를 불러와서 증폭을 해보자! > 적용한 걸 훈련시키자 > npy로 저장하자

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------------------------------------------------------------------------------
# 이미지 제너레이터 선언
train_datagen = ImageDataGenerator(
    rescale=1./255,             # 전처리/ 스케일링. 흑백이니까 1./255
    horizontal_flip=True,       # 수평 뒤집기
    vertical_flip=True,         # 수직 뒤집기
    width_shift_range=0.1,      # 수평 이동
    height_shift_range=0.1,     # 수직 이동
    rotation_range=5,           # 회전
    zoom_range=1.2,             # 확대
    shear_range=0.7,            # 층 밀리기 강도
    fill_mode='nearest'         # 빈자리는 근처에 있는 것으로(padding='same'과 비슷)
)

# 설명 참고
# https://keras.io/ko/preprocessing/image/

test_datagen = ImageDataGenerator(rescale=1./255)
# 트레인만 이미지 증폭을 하고 테스트를 할 때는 증폭할 필요가 없다. rescale은 전처리니까 같이 해주는 것 뿐
# 여기까지는 정의한 했고 이제 적용시켜보자 > flow 또는 flow_from_directory(옛날에는 폴더라는 이름이 없어서 디렉토리라고 썼다)

# -----------------------------------------------------------------------------------------------------
# 폴더(디렉토리)에서 불러와서 적용하기! fit과 같다. 이 줄을 지나면 x와 y가 생성이 된다.
# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',  
    target_size=(150,150), 
    batch_size=160, #대충 200으로 크게 줘도 된다. 
    class_mode='binary' 
)     
# 로그 > Found 160 images belonging to 2 classes.
# 예상 생성 사이즈 : ad > (80,150,150,3)

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',  
    target_size=(150,150), 
    batch_size=120,  
    class_mode='binary'  
) 
# 로그 > Found 120 images belonging to 2 classes.

# -----------------------------------------------------------------------------------------------------
print(xy_train[0][0].shape)         #(160, 150, 150, 3)
print(xy_train[0][1].shape)         #(160,)

# npy로 저장하자 -----------------------------------------------------------------------------------------------------
np.save('../data/image/brain/npy/keras66_train_x.npy', arr = xy_train[0][0])
np.save('../data/image/brain/npy/keras66_train_y.npy', arr = xy_train[0][1])
np.save('../data/image/brain/npy/keras66_test_x.npy', arr = xy_test[0][0])
np.save('../data/image/brain/npy/keras66_test_y.npy', arr = xy_test[0][1])
print('===== save complete =====')

x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')
print('===== load complete =====')

print(x_train.shape, y_train.shape)
# (160, 150, 150, 3) (160,)
print(x_test.shape, y_test.shape)
# (120, 150, 150, 3) (120,)