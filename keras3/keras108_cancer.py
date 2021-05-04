# 얘는 이중분류~~!!^^

import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=33)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# ---------------------------------------------------------------------
'''
model = ak.StructuredDataClassifier(
    overwrite=True,
    max_trials=3,         # epoch을 *n번 시도했다.
    loss = 'mse',         # 로스 지정해주자
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
filepath = 'C:/data/h5/autokeras/'
mc = ModelCheckpoint(filepath, save_best_only=True, verbose=1)

model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[stop, lr, mc])

results = model.evaluate(x_test, y_test)

print(results)

# -------------------------------------------------------
best_model = model.tuner.get_best_model()
best_model.save('C:/data/h5/autokeras/keras108.h5')
# [0.10879667848348618, 0.9824561476707458]
'''
# -------------------------------------------------------
from tensorflow.keras.models import load_model
l_model = load_model('C:/data/h5/autokeras/keras108.h5')

l_results = l_model.evaluate(x_test, y_test)
print(l_results)
# [0.10879667848348618, 0.9824561476707458]

print('==== done ====')
