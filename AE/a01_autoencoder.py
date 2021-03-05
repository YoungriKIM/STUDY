# encode : 암호로 만들다
# decode : 해독하다
# https://blog.keras.io/building-autoencoders-in-keras.html

# 지금까지는 x / y 가 서로 달랐는데. 오토인코더는 x -> □ -> x(=y) 이런 구조이다. 오토인코더 다음에 gan으로 넘어간다.

# ㅇㅇㅇㅇㅇ  > 원데이터
# ㅣㅣㅣㅣㅣ
#   ㅇㅇㅇ    > 압축하면서 노이즈, 잡음 제거(훈련도 한다)
# ㅣㅣㅣㅣㅣ
# ㅇㅇㅇㅇㅇ  > 다시 증폭
# ------------------------------------------------------------------------------------------------

# 오토인코더를 해보자
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784)/255
# 둘 다 상관 없다.

print(x_train[0])
print(x_test[0])

# 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)    # 아웃풋 레이어에는 sigmoid를 써야한다.(이미지/255. 하면서 0~1 사이만 있어서)
# decoded = Dense(784, activation='relu')(encoded)     # relu를 쓰면 0과1사이로 수렴을 안 하고 마이너스였던 부분은 손실되고 나머지는 무한으로 가니까

# 오토 인코더로 지정
autoencoder = Model(input_img, decoded)

autoencoder.summary()

# 컴파일 변경하면서 알아보자 --------------------------------------------------------------------------------
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# autoencoder.compile(optimizer='adam', loss='mse')
# mnist의 이미지는 어차피 0과 1사이라서 mse / binary 모두 상관없다.

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# loss: 0.0732 - acc: 0.0137 - val_loss: 0.0740 - val_acc: 0.0119
# acc가 낮을 수 밖에 없다.
# -------------------------------------------------------------------------------------------------------

autoencoder.fit(x_train, x_train, epochs=30, batch_size=256, validation_split=0.2)
# x,y 모두 x_train으로 지정한다.

decoded_img = autoencoder.predict(x_test)

# 오토인코더 적용을 확인해보자
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(18,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)   # 이미지 옆의 눈금선을 False
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()