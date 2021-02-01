import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input, Flatten, MaxPooling1D, Dropout, Reshape, SimpleRNN, LSTM, LeakyReLU, GRU, Conv2D, MaxPool2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.backend import mean, maximum
import os
import glob
import random
import tensorflow.keras.backend as K

model = Sequential()
model.add(Conv1D(96, 2, input_shape=(48,8), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(96, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(96, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(96))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(7))
model.add(Dense(1))
