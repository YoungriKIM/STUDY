# 저장한 csv 파일을 불러와보자

import numpy as np
import pandas as pd

# df = pd.read_csv('../data/csv/iris_sklearn.csv')
# print(df)
# 인덱스가 데이터로 같이 들어간 것을 볼 수 있음. 인덱스 칼럼이 0번째 라는 것을 명시해주자

# df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0)
# print(df)
# 헤더도 명시해주자

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)    # None, 0, 1 등을 넣어서 인덱스와 헤더를 지정할 수 있다.
print(df)
# default index_col = None / default header = 0 
