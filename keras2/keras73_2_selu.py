# 과제 : elu, selu, reaky relu 수식으로 만들기

# selu (scaled exponential linear unit)
# relu와 비슷하지만 음수 값에서 올라올 때 부드럽게 만들어 준다.

import numpy as np
import matplotlib.pyplot as plt

def selu(x, alp, l):
    return l*((x > 0)*x + (x <= 0)*(alp * np.exp(x) - alp))


x = np.arange(-5, 5, 0.1)
y = selu(x, 2, 1)

# 그래프 그려보자
plt.plot(x, y)
plt.grid()
plt.show()