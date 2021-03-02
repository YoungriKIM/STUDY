# vgg16을 불러와서 해체해보자

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

model = VGG16(weights='imagenet', include_top = False, input_shape=(32,32,3))   # false이면 내가 인풋쉐이프를 정할 수 있다. True일 경우 불러온 모델에 맞는 쉐잎으로 한다.
print(model.weights)


# -----------------------------------------------------------------------------------------------------
model.trainable=False # 제가 훈련 안 시킬고에요 가중치만 주세오 > 이 가져온 가중치에서 수정이 시작된다.
# 디폴트 값은 True이다.

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))
# _________________________________________________________________
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________
# 26
# 26

# 연산되는 레이어는 13개라는 뜻 bias 13 / weights 13
# 써머리 뽑아서 보면 param이 연산되는 레이어는 13개 밖에 없다.

# -----------------------------------------------------------------------------------------------------
model.trainable=True

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))
# _________________________________________________________________
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
# _________________________________________________________________
# 26
# 0