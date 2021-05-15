# 메모리 터질 때 위에 세줄 추가하면 메모리를 전부 안 써서 오래 걸리지만 돌아가기는 한다 

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)