# 탄젠트를 이해해보자 (lstm 내부에 들어가 있다)

# tanh
# -1 ~ 1
# sigmoid의 vanishing gradient의 문제는 남아 있지만 중심을 0점으로 옮겨 최적화 되었다.

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

# 그래프 그려보자
plt.plot(x, y)
plt.grid()
plt.show()