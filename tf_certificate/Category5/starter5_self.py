# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
# QUESTION
#
# Build and train a neural network to predict sunspot activity using
# the Sunspots.csv dataset.
#
# Your neural network must have an MAE of 0.12 or less on the normalized dataset
# for top marks.
#
# Code for normalizing the data is provided and should not be changed.
#
# At the bottom of this file, we provide  some testing
# code in case you want to check your model.

# Note: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure.


import csv
import tensorflow as tf
import numpy as np
import urllib

# DO NOT CHANGE THIS CODE
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_model():
    # url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    # urllib.request.urlretrieve(url, '../Study/tf_certificate/Category5/sunspots.csv')

    time_step = []
    sunspots = []

    with open('../Study/tf_certificate/Category5/sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))

    series = np.array(sunspots)
    print(series.shape) #(3235,)

    # DO NOT CHANGE THIS CODE
    # This is the normalization function
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

    print(time)       # [   0    1    2 ... 3232 3233 3234]
    print(time.shape) # (3235,)

    # The data should be split into training and validation sets at time step 3000
    # DO NOT CHANGE THIS CODE
    split_time = 3000

    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    valid_set = windowed_dataset(x_valid, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    # 모델 만들기
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D, LSTM, GRU, LeakyReLU, concatenate

    model = tf.keras.models.Sequential([
      # YOUR CODE HERE. Whatever your first layer is, the input shape will be [None,1] when using the Windowed_dataset above, depending on the layer type chosen
      Conv1D(filters = 400, kernel_size = 2, strides=1, padding = 'same', activation='relu', input_shape=[None,1]),
      # MaxPooling1D(pool_size=2),
      # Conv1D(400, 2, padding='same'),
      # Conv1D(200, 2, padding='same'),
      # Conv1D(200, 2, padding='same'),
      # MaxPooling1D(pool_size=2),
      Dense(16),
      Dense(16),
      tf.keras.layers.Dense(1)
    ])
    model.summary()

    #3. 컴파일, 핏
    model.compile(loss='mae', optimizer='adam')
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    stop = EarlyStopping(monitor='loss', patience=16, mode='min')
    reducelr = ReduceLROnPlateau(monitor='loss', patience=8, factor=0.5, verbose=1)
    hist = model.fit(train_set, epochs=100, batch_size=32, verbose=1, callbacks=[stop, reducelr])

    #4. 평가, 예측
    result = model.evaluate(valid_set, batch_size=32)
    print('mse: ', result)
    # ====================================================
    # mse:  0.0012294882908463478


    # PLEASE NOTE IF YOU SEE THIS TEXT WHILE TRAINING -- IT IS SAFE TO IGNORE
    # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
    # 	 [[{{node IteratorGetNext}}]]
    #

    # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")



# THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
# BEFORE UPLOADING YOU CAN DO IT WITH THIS
#def model_forecast(model, series, window_size):
#    ds = tf.data.Dataset.from_tensor_slices(series)
#    ds = ds.window(window_size, shift=1, drop_remainder=True)
#    ds = ds.flat_map(lambda w: w.batch(window_size))
#    ds = ds.batch(32).prefetch(1)
#    forecast = model.predict(ds)
#    return forecast


#window_size = # YOUR CODE HERE
#rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
#rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

#result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()

## To get the maximum score, your model must have an MAE OF .12 or less.
## When you Submit and Test your model, the grading infrastructure
## converts the MAE of your model to a score from 0 to 5 as follows:

#test_val = 100 * result
#score = math.ceil(17 - test_val)
#if score > 5:
#    score = 5

#print(score)


'''
# window_dataset 에 대하여 -------------------------------------
print('익스팬드 전', series)
# 익스팬드 전 [0.24284279 0.26192868 0.29306881 ... 0.03314917 0.03992968 0.00401808]
# shape=(3235,)
# csv에서 그냥 쭈욱 가져온 흑점에 대한 데이터

series = tf.expand_dims(series, axis=-1)
print('익스팬드 후',series)
# 익스팬드 후 tf.Tensor(
# [[0.24284279]
#  [0.26192868]
#  [0.29306881]
#  ...
#  [0.03314917]
#  [0.03992968]
#  [0.00401808]], shape=(3235, 1), dtype=float64)
# 리쉐잎 했다고 이해

ds = tf.data.Dataset.from_tensor_slices(series)
print('data.dataset 후:\n', list(ds))
# <tf.Tensor: shape=(1,), dtype=float64, numpy=array([0.24284279])
# (3235,1)짜리를 > (1,)*3235 개로 한 줄 씩 나눔

ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
# 한 칸씩 띄워서 31개씩 잘라서 반복하여 데이터로 만들음
# ex)            [0,1,2,3 ~ 30] 31개
#       1씩 차이 [1,2,3,4 ~ 31]
#   남긴거 버림   [3210 ~ 3235]
print('window 후\n: ', ds)
# :  <WindowDataset shapes: DatasetSpec(TensorSpec(shape=(1,), dtype=tf.float64, name=None), TensorShape([])), types: DatasetSpec(TensorSpec(shape=(1,), dtype=tf.float64, name=None), TensorShape([]))>
# 31개씩 1개 띄워서 배열을 만들어줌

ds = ds.flat_map(lambda w: w.batch(window_size + 1))
print('flat_map 후\n: ', ds)
# 위에서 만든 배열을 31개씩 한 세트로 붙여줌

ds = ds.shuffle(shuffle_buffer_size)
print('shuffle 후\n: ', ds)
# 데이터를 섞어줌

ds = ds.map(lambda w: (w[:-1], w[1:]))
# 각 줄에서 끝에 하루 뺀 것 하나 / 첫날 하루 뺀 것 하나 씩
# 즉 31에서 끝에 뺀 30 / 앞에 뺸 30

return ds.batch(batch_size).prefetch(1)
# 학습데이터를 세트채로 들고온다
# =======================-------------------------------------
'''




# PLEASE NOTE IF YOU SEE THIS TEXT WHILE TRAINING -- IT IS SAFE TO IGNORE
# BaseCollectiveExecutor::StartAbort Out of range: End of sequence
# 	 [[{{node IteratorGetNext}}]]
#

# YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
