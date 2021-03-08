# keras67_1 남자 여자에 잡음 넣고 복구하는 AE를 만들어라!
# npy 만든 거를 쓰면 빠르겠지?

import numpy as np
import matplotlib.pyplot as plt

x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
x_test = np.load('../data/image/gender/npy/keras67_test_x.npy')

print(x_train.shape)
print(x_test.shape)
# (200, 56, 56, 3)
# (200, 56, 56, 3)

# plt.imshow(x_test[10])
# plt.show()

# 여기까지는 동일 ------------------------------------------------------------------

# 노이즈를 만들어야겠지
x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min = 0, a_max= 1)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1)

# ---------------------------------------------------------------------------------
# ae 모델 만들자
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, BatchNormalization, LeakyReLU, Conv2DTranspose
from tensorflow.keras.layers import Dropout,Activation

def autoencoder():
    model = Sequential()
    model.add(Conv2D(512, 3, activation= 'relu', padding= 'same', input_shape = (56,56,3)))
    model.add(Conv2D(512, 5, activation= 'relu', padding= 'same'))
    model.add(Conv2D(3, 3, padding = 'same', activation= 'sigmoid'))

    return model

model = autoencoder()
model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train_noised, x_train, epochs = 20, batch_size=64)

# ---------------------------------------------------------------------------------
output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20,7))

# 이미지 다섯 개를 무작위로 고르자
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력 데이터) 이미지를 맨 위에 그리자
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(56,56,3))
    if i == 0:
        ax.set_ylabel('INPUT', size=10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(56,56,3))
    if i == 0:
        ax.set_ylabel('NOISE', size=10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더로 복원한 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(56,56,3))
    if i == 0:
        ax.set_ylabel('OUTPUT', size=10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
