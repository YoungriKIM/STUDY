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
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly

import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
    local_zip = 'testdata.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/testdata/')
    zip_ref.close()

    train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=(-1,1),
    height_shift_range=(-1,1),
    fill_mode='nearest',
    )
    #Your code here. Should at least have a rescale. Other parameters can help with overfitting.)

    validation_datagen = ImageDataGenerator(rescale=1./255)

    batch = 10
    train_dir = 'tmp/horse-or-human'
    train_generator = train_datagen.flow_from_directory(
        train_dir
        , target_size=(300, 300)
        , batch_size = batch
        , class_mode='binary'
    )

    test_dir = 'tmp/testdata'
    validation_generator = validation_datagen.flow_from_directory(
        test_dir
        , target_size=(300, 300)
        , batch_size = batch
        , class_mode='binary'
    )

    print(train_generator[0][0].shape, validation_generator[0][0].shape)
    #(10, 300, 300, 3) (10, 300, 300, 3)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dropout, Dense

    model = tf.keras.models.Sequential([
        # Note the input shape specified on your first layer must be (300,300,3)
        # Your Code here
        Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', input_shape=(300,300,3)),
        Conv2D(filters=128, kernel_size=3, activation='relu'),
        MaxPool2D(3,3),
        Conv2D(filters=64, kernel_size=3, activation='relu'),
        Conv2D(filters=64, kernel_size=3, activation='relu'),
        MaxPool2D(3,3),
        Flatten(),
        Dense(64, activation='relu'),
        # This is the last layer. You should not change this code.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import numpy as np
    stop = EarlyStopping(monitor='val_loss', patience=8, mode='min')
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, mode='min')
    
    model.fit(train_generator, epochs=10,
             validation_data=validation_generator,
             callbacks=[stop,lr])
    # steps_per_epoch=np.ceil(1072/batch) > 1072 = 말500 + 사람572
    # validation_steps=np.ceil(256/batch) > 256 = 말128 + 사람128

    result = model.evaluate(validation_generator)
    print('loss: ', result[0], '\nacc: ', result[1])

    return model

    # NOTE: If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

# ================================
# loss:  1.9158985614776611
# acc:  0.875