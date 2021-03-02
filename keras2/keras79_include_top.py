# 인클루드탑에 대한 내용

from tensorflow.keras.applications import VGG16

# 1)
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) # 크기 일부터 디폴트로 바꿔줌

# 2)
# model = VGG16()

model.trainable = False
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

# ----------------------------------------------------
# 1) 에서 include_top = False 하고 input_shape=(224, 224, 3)
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
# 26
# 0
# ----------------------------------------------------
# 2)
# Total params: 138,357,544
# Trainable params: 0
# Non-trainable params: 138,357,544
# 32
# 0
# ----------------------------------------------------
# 1) 에서 include_top = True 하고 input_shape=(224, 224, 3)
# Total params: 138,357,544
# Trainable params: 0
# Non-trainable params: 138,357,544
# 32
# 0