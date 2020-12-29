from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

inputs = Input(shape=(5,)) #input_dim=5 와 같은 의미
apple = Dense(5, activation='relu')(inputs)
banana = Dense(3)(apple)
tomato = Dense(4)(banana)
outputs = Dense(1) (tomato)
model = Model(inputs = inputs, outputs = outputs)
model.summary()
