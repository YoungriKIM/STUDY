# vgg16을 불러와서 해체해보자
# 윗 부분을 include_top=False로 바꿨으니 아래 쪽을 바꿔보자

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights='imagenet', include_top = False, input_shape=(32,32,3))   # false이면 내가 인풋쉐이프를 정할 수 있다. True일 경우 불러온 모델에 맞는 쉐잎으로 한다.
print(vgg16.weights)

vgg16.trainable=False

vgg16.summary()
print(len(vgg16.weights))                   # 26
print(len(vgg16.trainable_weights))         # 0

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
model.summary()
print('토탈 가중치의 수: ', len(model.weights))                                          # 26 > 32 (13개에서 16개. 3개는 내가 추가한 시퀀셜에 있는 것들)
print('동결(Freezen) 후 훈련되는 가중치의 수: ',len(model.trainable_weights))         # 0 > 6

#### 75_3에서 작성하는 내용 ####
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])

print(aaa)