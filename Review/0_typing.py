

# 랜덤값을 넣어 노이즈 추가
x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min = 0, a_max= 1)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1)



from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, BatchNormalization, LeakyReLU, Conv2DTranspose
from tensorflow.keras.layers import Dropout,Activation

def autoencoder():
    model = Sequential()
    model.add(Conv2D(512, 3, activation= 'relu', padding= 'same', input_shape = (150,150,3)))
    model.add(Conv2D(512, 5, activation= 'relu', padding= 'same'))
    model.add(Conv2D(3, 3, padding = 'same', activation= 'sigmoid'))

    return model


model = autoencoder()
model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train_noised, x_train, epochs = 10, batch_size=64)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
        plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다!!
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]])
    if i==0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]])
    if i==0:
        ax.set_ylabel('NOISE', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]])
    if i==0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()