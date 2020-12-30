#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 모델 1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation = 'relu')(input1)
dense1 = Dense(5, activation = 'relu')(dense1)

# 모델 2
input2 = Input(shape=(3,))
dense2 = Dense(1, activation = 'relu')(input2)
dense2 = Dense(2, activation = 'relu')(dense2)
dense2 = Dense(3, activation = 'relu')(dense2)
dense2 = Dense(4, activation = 'relu')(dense2)

# 모델 병합 / concatenate = 사슬처럼 엮다
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([dense1, dense2]) 
middle1 = Dense(5)(merge1)
middle1 = Dense(6)(middle1)
middle1 = Dense(7)(middle1) 

#모델 분기
output1 = Dense(8)(middle1)
output1 = Dense(9)(output1)
output1 = Dense(10)(output1) #얘네가 y1꺼

output2 = Dense(11)(middle1)
output2 = Dense(12)(output2)
output2 = Dense(13)(output2)
output2 = Dense(14)(output2) #얘네가 y2꺼

output3 = Dense(15)(middle1)
output3 = Dense(16)(output3)
output3 = Dense(17)(output3)
output3 = Dense(3)(output3) #얘네가 y3꺼

# 모델 선언
model = Model(inputs = [input1, input2], outputs = [output1, output2,output3])
model.summary()