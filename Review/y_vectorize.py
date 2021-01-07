import numpy as np

a = np.array([[3,6,5,4,2]])

print(a)            #[[3 6 5 4 2]]
print(a.shape)      #(5,1)

# sklearn - onhotencoder
from sklearn.preprocessing import OneHotEncoder
hot = OneHotEncoder()
hot.fit(a)
a = hot.transform(a).toarray()

print(a)            #[[1. 1. 1. 1. 1.]]
print(a.shape)      #(1, 5)
