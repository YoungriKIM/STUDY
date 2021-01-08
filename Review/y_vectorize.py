'''
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
'''

import numpy as np

z = np.array([3,6,5,4,2,9])

print(z)            #[3 6 5 4 2]
print(z.shape)      #(5,)

z = z.reshape(z.shape[0], 1)

from tensorflow.keras.utils import to_categorical
z = to_categorical(z)

print(z)
print(z.shape)