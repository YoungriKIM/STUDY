# 과제 : elu, selu, reaky relu 수식으로 만들기  

# LeakyReLU (leaky: 구멍이 난)
# relu에서 0보다 작은 값을 0으로 만들어 손실이 발생하니
# 0보다 작은 값은 0에 근접하는 매우 작은 값을 바꾸자
# 일반적으로 alpha 값으로 0.01을 준다.

import numpy as np
import matplotlib.pyplot as plt

def Leaky_ReLU(x, alp):
    return np.maximum(alp*x , x)

x = np.arange(-5, 5, 0.1)
y = Leaky_ReLU(x, 0.01)

# 그래프 그려보자
plt.plot(x, y)
plt.grid()
plt.show()