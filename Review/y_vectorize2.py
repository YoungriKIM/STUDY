import numpy as np

c = np.array([[3,6,5,4,2]])

print(c)            #[[3 6 5 4 2]]
print(c.shape)      #(1, 5)

c = np.transpose(c)

print(c)
                    # [[3]
                    #  [6]
                    #  [5]
                    #  [4]
                    #  [2]]
print(c.shape)      #(5, 1)

from sklearn.preprocessing import OneHotEncoder
hot2 = OneHotEncoder()
hot2.fit(c)
c = hot2.transform(c).toarray()

print(c)
                    # [[0. 1. 0. 0. 0.]
                    #  [0. 0. 0. 0. 1.]
                    #  [0. 0. 0. 1. 0.]
                    #  [0. 0. 1. 0. 0.]
                    #  [1. 0. 0. 0. 0.]]
print(c.shape)      #(5, 5)