# 준호가 알려준 이미지 전처리
# 이 파일을 기준으로 놓고 쓰겠삼

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import PIL.Image as pilimg

## 100개로 해보기 ## > 5만개 가자~~~

# x train 데이터 불러오기 -------------------------------------
df_pix = []
number = 50000

for a in np.arange(0, number):             
    file_path = '../Users/Admin/Desktop/dacon/dacon12/dirty_mnist_2nd/' + str(a).zfill(5) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_pix.append(pix)

x_df = pd.concat(df_pix)
x_df = x_df.values
# 원래 사이즈는 (256,256)

print(type(x_df))  # <class 'numpy.ndarray'>
print(x_df.shape)  # (25600, 256)


# 이미지 전처리 -------------------------------------

#254보다 작고 0이아니면 0으로 만들어주기
x_df2 = np.where((x_df <= 254) & (x_df != 0), 0, x_df)

# 이미지 팽창
x_df3 = cv2.dilate(x_df2, kernel=np.ones((2, 2), np.uint8), iterations=1)

# 블러 적용, 노이즈 제거
x_df4 = cv2.medianBlur(src=x_df3, ksize= 5)


# 이미지 리쉐잎 -------------------------------------
# 리쉐잎
x_dataset = x_df4.reshape(number, 256, 256, 1)

print(x_dataset.shape)  #(100, 256, 256, 1)

# npy로 저장 -------------------------------------
np.save('../data/npy/dacon12/dirty_mnist_train_all(50000).npy', arr=x_dataset)
print('===== save complete =====')

# npy로 저장 잘 되었나 확인 -------------------------------------
load_x = np.load('../data/npy/dacon12/dirty_mnist_train_all(50000).npy')
print('===== save complete =====')

print(load_x.shape)  #(100, 256, 256, 1)

# 100개 저장 파일 용량 > 6402Kb