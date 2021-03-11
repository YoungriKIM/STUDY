# 정규화, 초기값 설정기에 대한 정리


from tensorflow import keras
from tensorflow.keras.layers import Dense

model = model
# 1) =========================================================
# Initializers : :[컴퓨터]초기 내용을 설정하다
# 케라스 레이어의 초기 난수 가중치를 설정하는 방식을 규정

# ex)
model.add(Dense(64, kernel_initializer='random_uniform',
                        bias_initializer='zeros'))

# 변수지정: bias_initializer 바이어스용 / kernel_initializer 웨이트용

# 종류
# 'zeros' = 모든 값이 0인 텐서를 생성
# 'ones' = 모든 값이 1인 텐서를 생성
# 'constant(value=0)' = 모든 값이 특정 상수인 텐서를 생성

keras.initializers.Zeros()
# 의 형태로도 사용가능하다.

# https://keras.io/ko/initializers/

# 2) =========================================================
# regularizer : 레이어의 가중치 정규화
# 정규화를 사용하면 최적화 중에 레이어 매개 변수 또는 레이어 활동에 페널티를 적용 할 수 있다.

# kernel_regularizer: 레이어의 커널에 페널티를 적용하는 정규화(=weight)
# bias_regularizer: 레이어 바이어스에 페널티를 적용하는 레귤레이터
# activity_regularizer: 레이어 출력에 페널티를 적용하는 레귤레이터

# ex)
from tensorflow.keras import layers
from tensorflow.keras import regularizers

layer = layers.Dense(
    units=64,
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)
)

# 종류
# tf.keras.regularizers모듈의 일부로 다음과 같은 기본 제공 정규화를 사용할 수 있다.
from tensorflow.keras.regularizers import l1, l2, l1_l2

# https://keras.io/api/layers/regularizers/

# 3) =========================================================
# tf.keras.layers

from tensorflow.keras.layers import BatchNormalization, Dropout

# BatchNormalization : 입력을 정규화하는 레이어
# 배치 정규화는 평균 출력을 0에 가깝게 유지하고 출력 표준 편차를 1에 가깝게 유지하는 변환을 적용합니다.
# https://keras.io/api/layers/normalization_layers/batch_normalization/

# Dropout : 입력에 드롭 아웃을 적용
# 드롭 아웃 계층은 rate 훈련 시간 동안 각 단계에서 빈도를 사용하여 입력 단위를 무작위로 0으로 
# 설정하여 과적 합을 방지합니다. 0으로 설정되지 않은 입력은 모든 입력에 대한 합계가 변경되지
# 않도록 1 / (1-속도)만큼 확장됩니다.
# https://keras.io/api/layers/regularization_layers/dropout/
