# 과제 : elu, selu, reaky relu 수식으로 만들기

# elu (Exponential Linear Unit) Exponential: 기하급수적인
# -1 ~ ∞
# relu가 0이하 일 때 모두 0으로 변경해서 생기는 문제(Dying ReLU)를 막고자 나옴
# 일반적으로 alp(alpha) 값으로 1을 설정한다.

import numpy as np
import matplotlib.pyplot as plt

def elu(x, alp):
    return (x>0)*x + (x<=0)*(alp*(np.exp(x)-1))

x = np.arange(-5, 5, 0.1)
y = elu(x, 1)

# 그래프 그려보자
plt.plot(x, y)
plt.grid()
plt.show()

###
# TLDR : 어떠한 활성화 함수를 써야할까?
# 일반적으로 ELU → LeakyReLU → ReLU → tanh → sigmoid 순으로 사용한다고 한다.
# cs231n 강의에서는 ReLU를 먼저 쓰고 ,
# 그다음으로 LeakyReLU나 ELU 같은 ReLU Family를 쓰며, sigmoid는 사용하지 말라고 하고 있다.


