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
#
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

def solution_model():
    # url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    # urllib.request.urlretrieve(url, 'rps.zip')
    # local_zip = 'rps.zip'
    # zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('tmp/')
    # zip_ref.close()

    TRAINING_DIR = '../Study/tf_certificate/Category3/tmp/rps'
    training_datagen = ImageDataGenerator(
        # YOUR CODE HERE)
    rescale=1./255,
    width_shift_range=(-1,1),
    height_shift_range=(-1,1),
    fill_mode='nearest',
    validation_split=0.2
    )

    batch=16
    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR
        , target_size=(150,150)
        , batch_size=batch
        , class_mode='categorical'
        , subset='training'
    )

    test_generator = training_datagen.flow_from_directory(
        TRAINING_DIR
        , target_size=(150,150)
        , batch_size=batch
        , class_mode='categorical'
        , subset='validation'
    )

    print(train_generator[0][0].shape, test_generator[0][0].shape)
    # Found 2016 images belonging to 3 classes.
    # Found 504 images belonging to 3 classes.
    # (16, 150, 150, 3) (16, 150, 150, 3)

    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, Dense

    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
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

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(patience = 6)
    lr = ReduceLROnPlateau(factor = 0.25, verbose = 1, patience = 3)

    model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    model.fit_generator(train_generator, epochs = 1000, validation_data= test_generator,\
         steps_per_epoch= np.ceil(2016/32), validation_steps= np.ceil(504/32), callbacks = [es, lr])

    loss, acc = model.evaluate(test_generator)
    print("loss : ", loss)
    print("acc : ", acc)

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    # model.save("mymodel.h5")

# ============================
# loss :  0.11730863153934479
# acc :  0.954365074634552