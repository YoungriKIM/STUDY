# 이미지데이터를 불러와서 증폭을 해보자!

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
    '../data/image/brain/train',         # 이 안의 ad, normal 의 이미지를 전부 불러온다. ad 폴더는 x(80,150,150,1)가 되고 y는 0이 된다. / normal 폴더는 x(80,150,150,1)에 y는 1이 된다.
    target_size=(150,150),               # 이 크기로 불러온다.
    batch_size=5,                       # (전체 이미지 개수 / batch_size) = 한만큼의 배열이 생성된다.
    class_mode='binary'                  # binary로 했기 때문에 y를 0아니면 1로 해줄거고, 앞에 있는 폴더를 0으로 잡아준다.(ad가 normal보다 앞)
)     
# 로그 > Found 160 images belonging to 2 classes.
# 예상 생성 사이즈 : ad > (80,150,150,1)

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',      # 이 안의 ad, normal 의 이미지를 전부 불러온다. ad 폴더는 x(60,150,150,1)가 되고 y는 0이 된다. / normal 폴더는 x(60,150,150,1)에 y는 1이 된다.
    target_size=(150,150),           # 이 크기로 불러온다.
    batch_size=5,                    # (전체 이미지 개수 / batch_size)= 한만큼의 배열이 생성된다.
    class_mode='binary'              # binary로 했기 때문에 y를 0아니면 1로 해줄거고, 앞에 있는 폴더를 0으로 잡아준다.(ad가 normal보다 앞)
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

print(xy_train[15][1])       #[0. 0. 1. 1. 0. 0. 0. 1. 0. 1.] < 배치 10으로 한거임
# print(xy_train[16][1])     # ValueError: Asked to retrieve element 16, but the Sequence has length 16(16개있는데 17번째를 빼라라면 어떡해)
# batch_size를 10으로 했을 때 [0]~[15]까지 있다.(16개) 그래야 []하나 당 10개*16 = 160이니까
# batch_size가 5라면 총 160/5=32이니까 [0]~[31](32개)가 될 것이다.

# 그럼 batch_size를 160으로 주면? > [0].shape (160, 150, 150, 3) > [1]은 없겠지
# 그럼 batch_size를 260으로 주면? > 그래도 [0].shape가 (160, 150, 150, 3) 이상으로 나오지 않는다. > 역시 [1]은 없겠지
# 그럼 batch_size가 259으로 주면? > [0] > (159,150,150,3) / [1] > (1,150,150,3) 그럼 안 맞잖아. ㄱㅊ 어차피 우리는 통으로 뺄거라서 이렇게 자를 일이 없어서