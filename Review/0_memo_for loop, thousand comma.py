
# --------------------------
# for문: 지정한 값을 돌면서 넣어준다

import numpy as np

# aaa = range(4)

# for i in aaa:
#     print(i)

# --------------------------
# 결과값 천의자리 콤마 넣어서 보기

y_pred = [[89735.6]]
y_pred2 = np.array(y_pred)          # numpy로 
y_pred3 = float(y_pred2)            # 소수형으로
y_pred4 = format(y_pred3, ',')      # 천의자리 콤마

print(y_pred4)
