import numpy as np
from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

plt.imshow(x_train[1])
plt.show()