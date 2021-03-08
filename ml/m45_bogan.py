# 결측치를 없애는 방법들: 중간값, 평균값, 0으로, 위에 값, 아래 값, 행자체 삭제 등
# 그리고! 보간법이 있다.
# 어떤 방식이냐면... 데이터에서 nan 값을 빼고 모델을 만들어 훈련시킨다음 처음에 뺀 nan 값을 프레딕트에 넣는다.
# 이 기능도 이미 만들어져 있겠지?

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datastrs = ['3/1/2021', '3/2/2021', '3/3/2021', '3/4/2021', '3/5/2021']
dates = pd.to_datetime(datastrs)
print(dates)
# DatetimeIndex(['2021-03-01', '2021-03-02', '2021-03-03', '2021-03-04', '2021-03-05'],
# dtype='datetime64[ns]', freq=None)
print('======================================================================')

ts = Series([1, np.nan, np.nan, 8, 10], index = dates)
print(ts)
# 2021-03-01     1.0
# 2021-03-02     NaN
# 2021-03-03     NaN
# 2021-03-04     8.0
# 2021-03-05    10.0
# 날짜와 시계열데이터라서 nan 값이 1~8 사이에 있을 것 같지?

ts_intp_linear = ts.interpolate()   # interpolate : 말참견하다, 써넣음, 통계에서 사용
print(ts_intp_linear)
# 2021-03-01     1.000000
# 2021-03-02     3.333333
# 2021-03-03     5.666667
# 2021-03-04     8.000000
# 2021-03-05    10.000000

# 쉽고 빠르지만 이 기능을 절대적으로 생각하지 말고 결측치를 해결하는 모든 방법을 모두 사용해라!