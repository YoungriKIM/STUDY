import numpy as np

# a = np.array(range(1, 11))
# size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):     #행
        subset = seq[i : (i+size)]           #열
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

# dataset = split_x(a, size)
# print(dataset)

#(100,5) 를 원할 때
b = np.array(range(1, 105))
size = 5
b_dataset = split_x(b, size)

print(b_dataset.shape)

#(20,8) 를 원할 때
c = np.array(range(1, 28))
size = 8
c_dataset = split_x(c, size)

print(c_dataset.shape)

# 행이 1을 원하면?
d = np.array(range(1, 8))
size =7
d_dataset = split_x(d, size)

print(d_dataset.shape)


