from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1

model = EfficientNetB1()

model.trainable = False

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

# ===============================================
# 모델 별로 파라미터와 가중치 기록 할것
# ===============================================
# VGG16
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# 32
# 0

# VGG19
# Total params: 143,667,240
# Trainable params: 0
# Non-trainable params: 143,667,240
# 38
# 0

# Xception
# Total params: 22,910,480
# Trainable params: 0
# Non-trainable params: 22,910,480
# 236
# 0

# ResNet101
# Total params: 44,707,176
# Trainable params: 0
# Non-trainable params: 44,707,176
# 626
# 0

# ResNet101V2
# Total params: 44,675,560
# Trainable params: 0
# Non-trainable params: 44,675,560
# 544
# 0

# ResNet152
# Total params: 60,419,944
# Trainable params: 0
# Non-trainable params: 60,419,944
# 932
# 0

# ResNet152V2
# Total params: 60,380,648
# Trainable params: 0
# Non-trainable params: 60,380,648
# 816
# 0

# ResNet50
# Total params: 25,636,712
# Trainable params: 0
# Non-trainable params: 25,636,712
# 320
# 0

# ResNet50V2
# Total params: 25,613,800
# Trainable params: 0
# Non-trainable params: 25,613,800
# 272
# 0

# InceptionV3
# Total params: 23,851,784
# Trainable params: 0
# Non-trainable params: 23,851,784
# 378
# 0

# InceptionResNetV2
# Total params: 55,873,736
# Trainable params: 0
# Non-trainable params: 55,873,736
# 898
# 0

# MobileNet
# Total params: 4,253,864
# Trainable params: 0
# Non-trainable params: 4,253,864
# 137
# 0

# MobileNetV2
# Total params: 3,538,984
# Trainable params: 0
# Non-trainable params: 3,538,984
# 262
# 0

# DenseNet121
# Total params: 8,062,504
# Trainable params: 0
# Non-trainable params: 8,062,504
# 606
# 0

# DenseNet169
# Total params: 14,307,880
# Trainable params: 0
# Non-trainable params: 14,307,880
# 846
# 0

# DenseNet201
# Total params: 20,242,984
# Trainable params: 0
# Non-trainable params: 20,242,984
# 1006
# 0

# NASNetLarge
# Total params: 88,949,818
# Trainable params: 0
# Non-trainable params: 88,949,818
# 1546
# 0

# NASNetMobile
# Total params: 5,326,716
# Trainable params: 0
# Non-trainable params: 5,326,716
# 1126
# 0

# EfficientNetB0
# Total params: 5,330,571
# Trainable params: 0
# Non-trainable params: 5,330,571
# 314
# 0

# EfficientNetB1
# Total params: 7,856,239
# Trainable params: 0
# Non-trainable params: 7,856,239
# 442
# 0