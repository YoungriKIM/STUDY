# x의 범위를 지정한 것 처럼 y이도 범위를 지정하여 라벨링을 바꿀 수 있지 않을까????
# 그 전에 데이터를 해부해서 보자!

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten
import pandas_profiling

wine = pd.read_csv('../data/csv/winequality-white.csv',index_col=None, header=0, sep=';')

# groupby: 그룹별로 집계하는 기능
count_data = wine.groupby('quality')['quality'].count()
print(count_data)
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5
# Name: quality, dtype: int64

# 함 그려볼까
import matplotlib.pyplot as plt
count_data.plot()
plt.show()