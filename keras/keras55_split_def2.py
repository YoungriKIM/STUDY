# 스플릿 함수 두 번째
# def split_x(D, x_col, x_low, y_col, y_len)

import numpy as np

dataset = np.array([[1,2,3,4,5,6,7,8],[11,12,13,14,15,16,17,18], [21,22,23,24,25,26,27,28],[31,32,33,34,35,36,37,38],[41,42,43,44,45,46,47,48]])
dataset = np.transpose(dataset)
print(dataset.shape)    #(8, 5)
print(type(dataset))    #<class 'numpy.ndarray'>

aaa = dataset

def split_xy(aaa, x_row, x_col, y_row, y_col):
    x, y = list(), list()
    for i in range(len(aaa)):
        if i > len(aaa)-y_row:
            break
        tmp_x = aaa[i:i+x_row, :x_col]
        tmp_y = aaa[i:i+y_row, x_col:x_col+y_col]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy(dataset,4,3,5,2)
print(x, '\n\n', y)
print(x.shape)      #(4, 4, 3)
print(y.shape)      #(4, 5, 2)

