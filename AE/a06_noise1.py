# 노이즈를 없애보자!

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784)/255

# 노이즈를 만들어야겠지
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
#                                          -------- > 0~0.1사이를 찍어주면 원래 0~1이었던 데이터가 0~1.1이 된다.
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min =0, a_max=1)
#                                        ----------------- > 그래서  x_train_noised 안의 값을 0~1 사이로 제한하겠다.
x_test_noised = np.clip(x_test_noised, a_min =0, a_max=1)

# ae 모델 만들자
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units= 784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)
# 154의 의미: PCA 95% 적용할 때 값. 즉 가장 안정적인 95% 성능으로 확인하고자 함

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=10)
# 노이즈 있느 게 들어가서 없는 게 나오겠지?

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20,7))

# 이미지 다섯 개를 무작위로 고르자
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력 데이터) 이미지를 맨 위에 그리자
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더로 복원한 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()