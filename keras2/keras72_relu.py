# 렐루를 이해해보자

# relu (Rectified Linear Unit) rectified: 정류한(수정된)
# 0 ~ ∞
# 0보다 작으면 0, 크면 x/ 연산이 간결 해 학습 속도 빠름/큰 값이 1에 머무는 sigmoid문제 해결/
# vanishing gradient 문제 해결/ but 음수 값이 무조건 0 이 되면서 데이터 손실 가능성 존재

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

