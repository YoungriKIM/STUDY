import numpy as np
import matplotlib.pyplot as plt

def softmax(x) :
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1, 5)
y = softmax(x)

ratio = y
labels = y
plt.pie(ratio, labels = labels, shadow=True, startangle=90)     # 파이 그래프를 그려보자
plt.show()