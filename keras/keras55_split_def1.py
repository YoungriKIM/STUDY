# 스플릿 함수 첫 번째
# def split_x(D, x_len, y_len)

# x_len, y_len 기준

import numpy as np

dataset = np.array(range(1, 11))

def split_xy1(dataset, x_len, y_len):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + x_len
        y_end_number = x_end_number + y_len
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy1(dataset, 3, 2)
print(x, '\n\n', y)
print(x.shape)      # (6, 3)
print(y.shape)      # (6, 2)

# [[1 2 3]
#  [2 3 4]
#  [3 4 5]
#  [4 5 6]
#  [5 6 7]
#  [6 7 8]] 

#  [[ 4  5]
#  [ 5  6]
#  [ 6  7]
#  [ 7  8]
#  [ 8  9]
#  [ 9 10]]