import numpy as np

x = [1,2]
y = [1,2]

z = np.concat((x,y), axis=1)

print(z)

arr = np.array([
    [
        [1, 1],
        [2, 2]
    ],
    [
        [3, 3],
        [4, 4]
    ]
])


item = np.array([
    [5, 5],
    [6, 6]
])

print(arr.shape)
print(item.shape)

append = np.append(arr, item.reshape(1, 2, 2), axis=0)

print(append)