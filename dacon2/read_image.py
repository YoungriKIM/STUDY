import PIL.Image as pilimg
import numpy as np
import pandas as pd
 
# # Read image ==================================================
# image = pilimg.open('../Users/Admin/Desktop/dacon/mnist/dirty_mnist/00000.png')
 
# # Display image ==================================================
# # image.show()
 
# # Fetch image pixel data to numpy array ==================================================
# pix = np.array(image)
# print(pix.shape)        #(256, 256)


# 샘플 10개 불러오기 / x_train ===========================================================

# df_train = []

# for a in np.arange(0, 10000):
#     b = str(a)
#     c = str('0'*(5-len(b)))
#     i = (c+b)                                           ##################################### 불러와서 csv로 저장해서 편하게 쓰자
#     file_path = '../Users/Admin/Desktop/dacon/mnist/dirty_mnist/' + str(i) + '.png'
#     image = pilimg.open(file_path)
#     pix = np.array(image)
#     pix = pd.DataFrame(pix)
#     df_train.append(pix)

# x = pd.concat(df_train)
# x = x.values
# print(x.shape)       #(2560000, 256)

# x_df = pd.DataFrame(x)

# x_df.to_csv('../Users/Admin/Desktop/dacon/mnist/newsavetrain/0~9999.csv')
#------------------------------------------------------------------------------------

x_df = pd.read_csv('../Users/Admin/Desktop/dacon/mnist/newsavetrain/0~9999.csv', index_col=0, header=0)    # None, 0, 1 등을 넣어서 인덱스와 헤더를 지정할 수 있다.

print(x_df.info())
x = x_df.values

a = 10000
x = x.reshape(a, int(x.shape[0]/a), x.shape[1], 1)
print(x.shape)       #(10000, 256, 256, 1)

'''

# 샘플 10개 불러오기 / y_train ===========================================================

y_dataset = pd.read_csv('../Users/Admin/Desktop/dacon/mnist/dirty_mnist_answer.csv', index_col=None, header=0)
print(y_dataset.shape)      # (50000, 27)

y = y_dataset.iloc[:10, :]
print(y.shape)              # (10, 27)  # 다중 분류 모델

# 샘플 10개 불러오기 / x_test ===========================================================

df_test = []

for i in range(0,10):
    file_path = '../Users/Admin/Desktop/dacon/mnist/test_dirty_mnist/5000' + str(i) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_test.append(pix)

x_test = pd.concat(df_test)
x_test = x_test.values
print(x_test.shape)       #(2560, 256)

x_test = x_test.reshape(10, 256, 256, 1)
print(x_test.shape)       #(10, 256, 256, 1)

# 전처리 ==============================================================
x = x.reshape(10, 256*256*1)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

# 스탠다드 스케일러
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)

# 모델에 넣을 쉐잎으로 정리
x_train = x_train.reshape(8, 256, 256, 1)
x_val = x_val.reshape(2, 256, 256, 1)

# 모델 구성 ==============================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', strides=1, input_shape=(256,256,1)))
model.add(Conv2D(10, 2, padding='same'))
model.add(Flatten())
model.add(Dense(27, activation = 'softmax'))

# model.summary()

# 컴파일, 훈련 ==============================================================
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=8, validation_data=(x_val, y_val), verbose=1)

# 평가, 예측 =============================================================
loss, acc = model.evaluate(x_train, y_train, batch_size=8)
print('loss, acc: ', loss, acc)

y_pred = model.predict(x_test).round(2)
print(y_pred)


# 예측값을 submission에 넣기 ========================================
subfile = pd.read_csv('../Users/Admin/Desktop/dacon/mnist/sample_submission.csv')

pred_save = pd.DataFrame(y_pred)

# subfile.iloc[:10, :] = pred_save

print(subfile.head())

pred_save.to_csv('../Users/Admin/Desktop/dacon/mnist/submit/sample_submission_1.csv', index=False)
'''