import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, Flatten, MaxPooling2D, Dropout

#1. data
from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape)    #(50000, 32, 32, 3)
# print(y_train.shape)    #(50000, 1)

# print(np.min(x_train), np.max(x_train))    #0 255
# print(np.min(y_train), np.max(y_train))    #0 99


# preprocessing : 3) y vectorize /  2) x minmaxscaler / 1) x split / 

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

x_train = x_train.astype('float32')/255.
x_val = x_val.astype('float32')/255.
x_test = x_test.astype('float32')/255.

# print(np.min(x_train), np.max(x_train))     #0.0 1.0

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

#2. model

model = Sequential()
model.add(Conv2D(filters = 128, kernel_size=(2,2), strides=1, padding='same', input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add (Conv2D(128, 2))
model.add(Dropout(0.2))
model.add (Conv2D(96, 2))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(100, activation='softmax'))

#3. compile, fit

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')

model.fit(x_train, y_train, epochs=300, batch_size=64, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])

#4. evaluate, predict
result = model.evaluate(x_test, y_test, batch_size=64)
print('loss: ', result[0])
print('acc: ', result[1])

y_pred = model.predict(x_test[:5])
print('y_pred: \n', y_pred.argmax(axis=1))
print('y_test[:5]: \n', y_test[:5].argmax(axis=1))
