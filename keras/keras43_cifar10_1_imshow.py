from tensorflow.keras.datasets import cifar10   #10가지로 분류하는것. cifar100은 100개로 분류

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)

print(x_train[10])
print(y_train[:10])

import  matplotlib.pyplot as plt
plt.imshow(x_train[10], 'brg')      #rgb가 아니라 brg
plt.show()
