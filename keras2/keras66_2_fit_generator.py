# 이미지데이터를 불러와서 증폭을 해보자! > 적용한 걸 훈련시키자
# 튠 알아서 완성 시켜~
# 
# Question : 증폭 된 이미지가 들어갔다는 증거를 찾아봐라!
# Answer: 데이터셋 자체가 증폭되는 것이 아니라 매 에폭마다 내가 지정한 데이터(우리의 경우는 steps_per_epoch=32)의 수 만큼의 매번 다른 ImageDataGenerator된 이미지가 들어간다.
# 1 epoch > 100개의 변형된 이미지'
# 2 epoch > 100개의 변형된 이미지''
# 3 epoch > 100개의 변형된 이미지'''  ... 

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------------------------------
# 이미지 제너레이터 선언
train_datagen = ImageDataGenerator(
    rescale=1./255,             # 전처리/ 스케일링. 흑백이니까 1./255
    # horizontal_flip=True,       # 수평 뒤집기
    # vertical_flip=True,         # 수직 뒤집기
    width_shift_range=0.1,      # 수평 이동
    height_shift_range=0.1,     # 수직 이동
    # rotation_range=5,           # 회전
    # zoom_range=1.2,             # 확대
    # shear_range=0.7,            # 왜곡
    # fill_mode='nearest'         # 빈자리는 근처에 있는 것으로(padding='same'과 비슷)
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

# 훈련을 시켜보자! 모델구성 -----------------------------------------------------------------------------------------------------
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(150,150,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit_generator(            # 파라미터 알아서 찾아봐
    xy_train,                   # fit은 x와 y를 따로 넣고, 핏제내레이터는 뭉쳐서 넣는다.
    steps_per_epoch=32,         # 지금 위 flow에서 batch_size를 5로해서 160/5 = 32 > 이 줄은 나눈 값을 넣는데, 크면 에러나고 작으면 손해를 본다.
    epochs=25,
    validation_data=xy_test,
    validation_steps=4      
)
# 이렇게 있겠지
# model.evaluate_generator
# model.predict_generator

# fit에서 history 반환 받아서 써먹기 ----------------------------------------------------------------------------------------------
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 시각화 하기
plt.subplot(2,1,1)
plt.plot(acc)
plt.plot(val_acc)
plt.title('acc, val_acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.subplot(2,1,2)
plt.plot(loss)
plt.plot(val_loss)
plt.title('loss, val_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])

plt.show() 

# history 받아서 이렇게 acc를 확인 할 수도 있다. -----------------------------------------------------------------------
print('acc: ', acc[-1])
# acc:  0.949999988079071

print('val_acc: ', val_acc[:-1])    # 전체가 다 나온다