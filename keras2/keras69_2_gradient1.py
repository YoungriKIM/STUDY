# 람다를 2차함수에 적용해보자

import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
x = np.linspace(-1, 6, 100) # -1부터 6까지를 100개로 넣겠다.
y = f(x)

# 그래프로 그려서 보자
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')    # 2,2 에만 점이 하나가 찍히겠지
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 점이 찍힌 지점(최적의 w)를 찾아보자