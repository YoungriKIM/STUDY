# pandas를 csv로 저장해보자

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()

x = dataset.data
# x = dataset['data']       >  이것도 가능하다.
y = dataset.target
# y = dataset['target']     >  이것도 역시 가능

# 판다스의 데이터프레임 지정
df = pd.DataFrame(x, columns=dataset.feature_names)

# 칼럼명이 너무 기니까 수정해보자
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  

# y 칼럼을 추가
df['Target'] = y

df.to_csv('../data/csv/iris_sklearn.csv', sep=',')      #seperate 나눠주겠다 , 로
# csv 파일을 test 폴더에 넣어주었는데 저렇게 색을 다르게 보려면 좌측의 확장>마켓플레이스> csv검색 > 레인보우 csv 다운
# 똑같이 edit csv를 다운 받으면 오른쪽 상단에 edit as csv가 뜨는데 클릭해서 수정도 가능하고 보기 편하다
