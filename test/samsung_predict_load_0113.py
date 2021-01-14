import numpy as np

x_train = np.load('../data/npy/samsung_x_train.npy')
y_train = np.load('../data/npy/samsung_y_train.npy')
x_val = np.load('../data/npy/samsung_x_val.npy')
y_val = np.load('../data/npy/samsung_y_val.npy')
x_test = np.load('../data/npy/samsung_x_test.npy')
y_test = np.load('../data/npy/samsung_y_test.npy')
x_pred = np.load('../data/npy/samsung_x_pred.npy')


from tensorflow.keras.models import load_model
model = load_model('../data/modelcheckpoint/samsung_14-891193.4375.hdf5')

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=1)
print('mse: ', result[0])
print('mae: ', result[1])

y_pred = model.predict(x_pred)
print('1/14일 삼성주식 종가: ', y_pred)

# mse:  1286656.875
# mae:  825.32763671875
# 1/14일 삼성주식 종가:  [[90572.59]]