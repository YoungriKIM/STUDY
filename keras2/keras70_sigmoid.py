# sigmoid 의 역할을 그림으로 그려보자

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print(x)
print(y)

# 그래프 그려보자
plt.plot(x, y)
plt.grid()
plt.show()