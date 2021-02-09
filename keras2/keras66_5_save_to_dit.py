# 이미지데이터를 불러와서 증폭을 해보자! > 저엉말로 증폭을 해서 저장해보자
# 37번째 줄이 주요 키워드!

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
    shear_range=0.7,            # 왜곡
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
    batch_size=5,                       # batch_size(5) * 아래 변수 건드린 횟수(나는 5) = 생성되는 이미지 개수(25개) > 이렇게 증폭이 가능
                                        # batch_size의 최대수는 160인데, 200처럼 더 큰 수를 지정해도 알아서 160(최대) * 변수건드린횟수 로 만들어 진다.
    class_mode='binary'         
    , save_to_dir='../data/image/brain_generator/train'     # ※caution※ 아래 찍어보자 부분처럼 변수를 한 번 건드려주어야 잘 저장이 된다!
)    
# 로그 > Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',    
    target_size=(150,150),     
    batch_size=5,              
    class_mode='binary'         
) 
# 로그 > Found 120 images belonging to 2 classes.


# 잘 생성이 됐는지 찍어보자 ------------------------------------------------------------------------
print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001E5D9A88550>

print(xy_train[0])
# x와 y를 모두 가지고 있다. 그런데 왜 5개지? batch_size를 5로 해서!, [0][0]은 x [0][1]은 y

print(xy_train[0][0]) # x만 나온다!
print(xy_train[0][0].shape)                        #(5, 150, 150, 3)

print(xy_train[0][1]) # y만 나온다!    
print(xy_train[0][1].shape)  #[1. 1. 0. 0. 1.]    #(5,)
