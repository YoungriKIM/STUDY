##=======================Add Td, T-Td and GHI features
def Add_features(data):
  c = 243.12
  b = 17.62
  gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
  dp = ( c * gamma) / (b - gamma)
  data.insert(1,'Td',dp)
  data.insert(1,'T-Td',data['T']-data['Td'])
  data.insert(1,'GHI',data['DNI']+data['DHI'])
  return data

  x = []
for i in quantiles:
    filepath_cp = f'../dacon/data/modelcheckpoint/dacon_06_y1_quantile_{i:.1f}.hdf5'
    model = load_model(filepath_cp, compile = False)
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    pred = pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred)
df_temp1 = pd.concat(x, axis = 1)
df_temp1[df_temp1<0] = 0
num_temp1 = df_temp1.to_numpy()
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = num_temp1