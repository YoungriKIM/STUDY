# 탄젠트를 이해해보자 (lstm 내부에 들어가 있다)

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

# 그래프 그려보자
plt.plot(x, y)
plt.grid()
plt.show()