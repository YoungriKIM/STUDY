import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,0], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

print(x.shape, y.shape, x_pred.shape) #(13, 3) (13,) (3,)
x = x.reshape(13,3,1)
x_pred =x_pred.reshape(1,3,1)
print(x.shape, y.shape, x_pred.shape) #(13, 3, 1) (13,) (1, 3, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
