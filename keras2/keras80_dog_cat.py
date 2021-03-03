# C:\data\image\vgg 에 개,고양이,라이언,슈트 사진 저장
# 파일명: dog1.jpg  /   cat1.jpg    /   lion.jpg    /   suit.jpg

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

#1. 데이터
img_dog = load_img('../data/image/vgg/dog1.jpg', target_size=(224,224))
img_cat = load_img('../data/image/vgg/cat1.jpg', target_size=(224,224))
img_lion = load_img('../data/image/vgg/lion1.jpg', target_size=(224,224))
img_suit = load_img('../data/image/vgg/suit1.jpg', target_size=(224,224))

# plt.imshow(img_dog)
# plt.show()
# 잘 나오는 것 확인

# 불러와서 써볼까
print(img_dog)
# <PIL.Image.Image image mode=RGB size=224x224 at 0x173CEFE8070>
# 형식을 바꿔줘야 하네?

# 어레이로 바꾸자
arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)
print(arr_dog)
#   [130. 108.  84.]
#   [133. 111.  90.]
#   [109.  87.  73.]]]
print(type(arr_dog))    # <class 'numpy.ndarray'>
print(arr_dog.shape)    # (224, 224, 3)         # vgg16의 디폴트가 224라서 맞춰준 것이다.
# 그런데 이 이미지의 형태는 RGB이다. > vgg에 넣을 때는 BGR로 바꿔야 한다.

# RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)
print(arr_dog)      # 형식이 바뀌었다.
#   [ -19.939003    -8.778999     6.3199997]
#   [ -13.939003    -5.7789993    9.32     ]
#   [ -30.939003   -29.779      -14.68     ]]]
print(arr_dog.shape)    # (224, 224, 3)

# 4개의 이미지를 순서대로 합쳐서 4차원으로 만들어야 하겠지?
arr_input = np.stack([arr_dog, arr_cat, arr_lion, arr_suit])
print(arr_input.shape)  # (4, 224, 224, 3)

#2. 모델 구성
model = VGG16()
results = model.predict(arr_input)

print(results)
# [[1.2521125e-08 1.9053208e-09 1.8513767e-09 ... 2.7165741e-09
#   7.1840458e-07 2.6765638e-06]
#  [1.8639266e-06 1.8009090e-05 1.0514532e-05 ... 1.8485185e-06
#   2.6443289e-05 4.2969445e-04]
#  [1.4814070e-06 1.2876735e-05 1.7772716e-06 ... 1.9167433e-06
#   3.0861062e-05 2.6065073e-04]
#  [5.9128644e-07 7.5139332e-07 1.8915797e-07 ... 4.8401833e-10
#   1.3194615e-07 7.9635742e-05]]
print('results.shape: ', results.shape)
# results.shape:  (4, 1000)
#                     ---- >> 이미지넷에서 분류할 수 있는 카테고리 총 1000개

# 결과값이 무슨 의미인지 확인하자
from tensorflow.keras.applications.vgg16 import decode_predictions      # 예측한 것을 해석하겠다.
decode_results = decode_predictions(results)
print('------------------------------------------')
print('dog.jpg는 : ', decode_results[0])
print('------------------------------------------')
print('cat.jpg는: ', decode_results[1])
print('------------------------------------------')
print('lion.jpg는: ', decode_results[2])
print('------------------------------------------')
print('suit.jpg는: ', decode_results[3])
print('------------------------------------------')

# 예측 결과
# ------------------------------------------
# dog.jpg는 :  [('n02099601', 'golden_retriever', 0.8731191), ('n02099712', 'Labrador_retriever', 0.052346446), ('n02111500', 'Great_Pyrenees', 0.051195905), ('n02104029', 'kuvasz', 0.012581478), ('n02102318', 'cocker_spaniel', 0.0021044335)]
# ------------------------------------------
# cat.jpg는:  [('n02123597', 'Siamese_cat', 0.1087263), ('n02883205', 'bow_tie', 0.08876292), ('n02085620', 'Chihuahua', 0.08649316), ('n02123045', 'tabby', 0.067285694), ('n03026506', 'Christmas_stocking', 0.059777938)]  
# ------------------------------------------
# lion.jpg는:  [('n03291819', 'envelope', 0.33172616), ('n03642806', 'laptop', 0.07295863), ('n04548280', 'wall_clock', 0.07267735), ('n03595614', 'jersey', 0.06543891), ('n02834397', 'bib', 0.04869516)]
# ------------------------------------------
# suit.jpg는:  [('n06359193', 'web_site', 0.8707195), ('n03594734', 'jean', 0.077155255), ('n04350905', 'suit', 0.009567844), ('n03877472', 'pajama', 0.008118273), ('n04370456', 'sweatshirt', 0.006131437)]
# ------------------------------------------