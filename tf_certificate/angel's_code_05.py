
import csv
import tensorflow as tf
import numpy as np
import urllib
import math
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'sunspots.csv')

    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))

    series = sunspots

    # DO NOT CHANGE THIS CODE
    # This is the normalization function
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

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
    validation_set = windowed_dataset(x_valid, window_size=window_size, batch_size=batch_size,
                                      shuffle_buffer=shuffle_buffer_size)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=[None, 1]),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
      # YOUR CODE HERE. Whatever your first layer is, the input shape will be [None,1] when using the Windowed_dataset above, depending on the layer type chosen
      tf.keras.layers.Dense(1)
    ])
    # PLEASE NOTE IF YOU SEE THIS TEXT WHILE TRAINING -- IT IS SAFE TO IGNORE
    # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
    # 	 [[{{node IteratorGetNext}}]]
    #

    model.summary()

    es = EarlyStopping(patience = 8)
    lr = ReduceLROnPlateau(patience=4, factor = 0.25, verbose = 1)
    model.compile(loss = 'mae', optimizer = 'adam')
    model.fit(train_set, epochs = 1000, validation_data= validation_set, callbacks = [es, lr])
    model.evaluate(validation_set)
    # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
    def model_forecast(model, series, window_size):
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(32).prefetch(1)
        forecast = model.predict(ds)
        return forecast


    window_size = 30
    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

    result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    print(result)

    # To get the maximum score, your model must have an MAE OF .12 or less.
    # When you Submit and Test your model, the grading infrastructure
    # converts the MAE of your model to a score from 0 to 5 as follows:

    test_val = 100 * result
    score = math.ceil(17 - test_val)
    if score > 5:
        score = 5

    print(score)

    # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
    return model