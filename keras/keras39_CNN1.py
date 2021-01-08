# keras.io 들어가서 Conv2D 레이어의 (10, (3,3), input_shape=(5,5,1)) 의 각자 이름 

# Conv2D(10, (3,3), input_shape = (N, 5, 5, 1))
# Conv2D(filters, kernel_size, input_shape = (batch_size, rows, cols, channels))
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters = 10, kernel_size=(2,2), strides=1, padding='same', input_shape=(10,10,1))) #CNN은 4차원 인풋 받아서 아웃풋도 4차원 / LSTM은 3차원 받아서 2차원
#  padding = 모서리 부분이 손해되니 한 번 감싸주겠다. 어떤걸로? 'same', 내가 준 이미지랑 같은 걸로 / 디폴트는 'valid'이다.
#  padding 의 목적은 다음으로 넘길 때 같은 크기로 주고 싶어서이다. 10,10이 인풋이고 원래 넘길 때는 9,9인데 패딩을 적용해서 10,10이 된다.
#  strides = 몇칸씩 가느냐. 자르고 나서 몇 칸 띄워서 또 자를 건지. 디폴트는 1이다. (1,2)등도 가능하다.
model.add(MaxPooling2D(pool_size=2)) #Maxpooling은 꼭 Conv 다음에 써야한다.
#  가로세로를 풀링사이즈로 나눠서 그 중 제일 큰 값을 남김
#  pool_size=2 디폴트가 2(2, 2), 3으로하면 (3,3), (2,3)도 가능하다.
model.add(Conv2D(9, (2,2), padding='valid'))
# model.add(Conv2D(9, (2,3)))
# model.add(Conv2D(8, 2))            # 2를 (2,2로 인식) #Conv2D는 특성을 증폭하여 추출하는 방식이니 복수로 하는 것이 좋다.
model.add(Flatten())
model.add(Dense(1))

model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 9, 9, 10)          50           > (input_dim * kernel_size + bias(1)) * filter > (1 * 2  * 2 + 1) * 10
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 8, 8, 9)           369          > (10 * 2 * 2 + 1) * 9 
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 7, 7, 8)           296
# _________________________________________________________________
# flatten (Flatten)            (None, 392)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 393
# =================================================================
# Total params: 1,108
# Trainable params: 1,108
# Non-trainable params: 0
# _________________________________________________________________
