
b = np.array([3,6,5,4,2])

print(b)            #[3 6 5 4 2]
print(b.shape)      #(5,)

b = b.reshape(b.shape[0], 1)

print(b.shape)      #(5, 1)

# tensorflow - to_categorical
from tensorflow.keras.utils import to_categorical
b = to_categorical(b)

print(b)
# [[0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0.]]
print(b.shape) (5, 7)