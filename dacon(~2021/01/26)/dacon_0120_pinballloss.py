from tensorflow.keras.backend import mean, maximum

def quantile_loss(q, y_test, y_predict):
  err = (y_test-y_predict)
  return mean(maximum(q*err, (q-1)*err), axis=-1)


model.compile(loss=lambda y_test,y_predict: quantile_loss(0.5,y_test,y_predict))#, **param)

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for q in q_lst:
  model.add(Dense(10))
  model.add(Dense(1))   
  model.compile(loss=lambda y,pred: quantile_loss(q,y,pred), optimizer='adam')
  model.fit(x,y, epoch=300)
