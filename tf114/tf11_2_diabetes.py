# 회귀모델데이터셋 불러와서 만들기

from sklearn.datasets import load_diabets
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

# [실습] 만들어라! /  결과는 sklearn의 r2_score/accuracy_score를 사용할 것