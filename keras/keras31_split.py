#이 방법을 통해 시계열 데이터를 받으면 x와 y를 나눌 수 있다.

import numpy as np

a = np.array(range(1, 11))
size = 5


print(len(a)) #10

'''
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

'''
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):     #행
        subset = seq[i : (i+size)]           #열
        aaa.append(subset) #이렇게 item for item in을 빼도 되는데 리스트가 한번 더 씌워져서 나온다.
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print('=====================')
print(dataset)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]



b = np.array(range(4,13))
dataset2 = split_x(b, size)
print(dataset2)
# [[ 4  5  6  7  8  9 10 11]
#  [ 5  6  7  8  9 10 11 12]]