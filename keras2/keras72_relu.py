# 렐루를 이해해보자

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x) # 0 보다 작은 때는 0 , 0보다 클 때 x 값이 그대로 간다

x = np.arange(-5, 5, 0.1)
y = relu(x)

# 그래프 그려보자
plt.plot(x, y)
plt.grid()
plt.show()

# 과제 : elu, selu, reaky relu 수식으로 만들기