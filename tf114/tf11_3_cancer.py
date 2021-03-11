# 이진분류 모델 데이터셋 불러와서 만들기
from sklearn.datasets import load_breast_cancer
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, ])
y = tf.placeholder(tf.float32, shape=[None, ])

# [실습] 만들어라! /  결과는 sklearn의 r2_score/accuracy_score를 사용할 것