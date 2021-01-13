import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
# print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

# print(dataset.values())

# print(dataset.target_names)

x = dataset.data
y = dataset.target

df = pd.DataFrame(x, columns=dataset.feature_names)

# print(df)
# print(df.describe())
# print(df.columns)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# print(df.describe())

df['Ydata'] = dataset.target
print(df.head())

print(df.isnull())
print(df.isnull().sum())

print(df['Ydata'].value_counts())

print(df.corr())


print(df2.shape)
x = df2.iloc[1740:2400, [1,2,3,5,6,7,8,9,10,11,12,13]]
# x = df2.iloc[1740:,[1,2,3,5,6,7,8,9,10,11,12,13,14]]
print(x)

# y = df2.iloc[1740:-1, 4]
