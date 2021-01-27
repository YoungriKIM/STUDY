# 사인 함수 함 그려보자

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1) # 0부터 9까지 0.1단위로 해서 100개
y= np.sin(x)    # 사인함수

plt.plot(x, y)
plt.show()