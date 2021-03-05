# 4번 복사해서 씀
# 첫번째 레이어는 무조건 COnvolution 중간중간은 맘대루
# 모델 더 딥~하게 구성할 것 ※원칙대로 수렴 후 원상태로 증폭 / 내 맘대로※ 2가지로 만들어
# 비교까지 해라

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_train_out = x_train.reshape(60000, 784).astype('float32')/255

x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

# 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Conv2DTranspose, LeakyReLU, Dropout, UpSampling2D, BatchNormalization, MaxPooling2D

# 컨볼루션 모델을 만들어보자~ ------------------------------------------------------------------------
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, strides=1, padding = 'same', input_shape=(28,28,1)))
    model.add(MaxPooling2D(2))
    # model.add(BatchNormalization())
    model.add(Conv2D(16, 2, 1))
    model.add(UpSampling2D((2, 2))) # 맥스풀 했으니 업 해줘야 한다.
    model.add(Flatten())
    model.add(Dense(units=hidden_layer_size/2))
    model.add(Dense(units=hidden_layer_size))
    model.add(Dense(units=784, activation='sigmoid'))
    return model
# ------------------------------------------------------------------------------------------------

model = autoencoder(hidden_layer_size=154)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train, x_train_out, epochs=10)

# 그림으로 그려보자
output = model.predict(x_test)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
