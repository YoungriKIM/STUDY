# pandas의 데이터셋을 np로 저장해보자

import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)    # None, 0, 1 등을 넣어서 인덱스와 헤더를 지정할 수 있다.
print(df)
# default index_col = None / default header = 0

print(df.shape)     #(150, 5)
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 150 entries, 0 to 149
# Data columns (total 5 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   sepal_length  150 non-null    float64
#  1   sepal_width   150 non-null    float64
#  2   petal_length  150 non-null    float64
#  3   petal_width   150 non-null    float64
#  4   Target        150 non-null    int64
# dtypes: float64(4), int64(1)
# memory usage: 7.0 KB
# None

# pandas를 넘파이로 바꾸는 방법

aaa = df.to_numpy()
print(aaa)
print(type(aaa))    #<class 'numpy.ndarray'>
# target 값이 1. 2. 등의 float으로 바뀌었는데 numpy는 한가지 자료형만 쓸 수 있기 때문이다.

bbb = df.values
print(bbb)
print(type(bbb))   #<class 'numpy.ndarray'>      둘은 같다. 알아서 쓰도록

np.save('../data/npy/iris_sklearn.npy', arr = aaa)      # 폴더에 가서 생성되었음을 확인함
