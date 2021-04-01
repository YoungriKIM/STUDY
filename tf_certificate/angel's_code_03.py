import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

import urllib.request
import zipfile
import tensorflow as tf
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def solution_model():
    # url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    # urllib.request.urlretrieve(url, 'rps.zip')
    # local_zip = 'rps.zip'
    # zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('C:/data/image/')
    # zip_ref.close()


    TRAINING_DIR = "C:/data/image/rps/"
    train_datagen = ImageDataGenerator(
        width_shift_range = 0.1,
        height_shift_range= 0.1,
        rescale = 1/255.,
        validation_split= 0.2
    )

    train_generator = train_datagen.flow_from_directory(
        directory = TRAINING_DIR,
        target_size = (150,150),
        class_mode = 'categorical',
        batch_size = 32,
        subset = 'training'
    )

    test_generator = train_datagen.flow_from_directory(
        directory = TRAINING_DIR,
        target_size = (150,150),
        class_mode = 'categorical',
        batch_size = 32,
        subset = 'validation'
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'valid', input_shape = (150, 150, 3)),
        tf.keras.layers.MaxPooling2D(3,3),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'valid'),
        tf.keras.layers.MaxPooling2D(3,3),
        tf.keras.layers.Conv2D(128, (5,5), activation = 'relu', padding = 'valid'),
        tf.keras.layers.MaxPooling2D(5,5),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    es = EarlyStopping(patience = 6)
    lr = ReduceLROnPlateau(factor = 0.25, verbose = 1, patience = 3)

    model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    model.fit_generator(train_generator, epochs = 1000, validation_data= test_generator,\
         steps_per_epoch= np.ceil(2016/32), validation_steps= np.ceil(504/32), callbacks = [es, lr])

    print(model.evaluate(test_generator, steps = np.ceil(504/32)))


    return model
