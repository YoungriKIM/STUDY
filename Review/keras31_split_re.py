import numpy as np

#  x 행 100, 열5 / y는 열1

kaka = np.array(range(1,106))
size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(kaka, size)

print(dataset.shape) #(100, 6)
print(dataset)
x = dataset[:, :-1]
y = dataset[:, -1]

print(x.shape) #(100,5)
print(y.shape) #(100,)

'========================================='

# x 11행에 70열 / y 30열

meow = np.array(range(1, 111))
size = 100

dataset2 = split_x(meow, size)
print(dataset2.shape) #(11, 100)

x2 = dataset2[:, 0:70] #(11, 70)
y2 = dataset2[:, 70:100] #(11,30)

print(x2.shape)
print(y2.shape)