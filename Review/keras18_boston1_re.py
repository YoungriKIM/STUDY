#데이터 제공 (보스턴 주택가격 데이터셋 사용)
import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) #(506, 13)
print(y.shape) #(506,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state=311)

#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

