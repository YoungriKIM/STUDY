import numpy as np
import pandas as pd

##=======================Add Td, T-Td and GHI features
def Add_features(data):
    
  gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)

  data.insert(1,'GHI',data['DNI']+data['DHI'])
  return data

train = Add_features(train)
X_test = Add_features(X_test)

df_train = train.drop(['Day','Minute'],axis=1)
df_test  = X_test.drop(['Day','Minute'],axis=1)

column_indices = {name: i for i, name in enumerate(df_train.columns)}

def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

def preprocess_data(data, is_train = True):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]

#===================================================================
# 예측값을 제출 형식에 넣기 (예측한 값 9컬럼에 다 복붙함)
sub = pd.read_csv('../data/DACON_0126/sample_submission.csv')

for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day7"), column_name] = y_pred[:,0]
for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day8"), column_name] = y_pred[:,1]

sub.to_csv('../data/DACON_0126/submission_0120_1.csv', index=False)


#===================================================================
# 내일!!
x = []
for i in quantiles:
    model = mymodel()
    filepath_cp = f'../dacon/data/modelcheckpoint/dacon_y1_quantile_{i:.1f}.hdf5'
    cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    model.fit(x_train,y1_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y1_val),callbacks = [es,cp,lr])
    pred = pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred)
df_temp1 = pd.concat(x, axis = 1)
df_temp1[df_temp1<0] = 0
num_temp1 = df_temp1.to_numpy()
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = num_temp1

x = []
# 모레!!
for i in quantiles:
    model = mymodel()
    filepath_cp = f'../dacon/data/modelcheckpoint/dacon_y2_quantile_{i:.1f}.hdf5'
    cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    model.fit(x_train,y2_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y2_val),callbacks = [es,cp,lr])
    pred = pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred)
df_temp2 = pd.concat(x, axis = 1)
df_temp2[df_temp2<0] = 0
num_temp2 = df_temp2.to_numpy()
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = num_temp2
        
submission.to_csv('./practice/dacon/data/111.csv', index = False)

#===================================================================

sub = pd.read_csv('../data/DACON_0126/sample_submission.csv')

for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day7"), column_name] = y_pred[:,0]
for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day8"), column_name] = y_pred[:,1]

sub.to_csv('../data/DACON_0126/submission_0120_1.csv', index=False)