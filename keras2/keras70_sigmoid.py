# sigmoid 의 역할을 그림으로 그려보자

# sigmiod
# 0 ~ 1
# 입력에 대해 무조건 0과 1 사이로 변환, 층을 거듭할 수록 값이 작아지는 문제(vanishing gradient)가 발생한다.

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