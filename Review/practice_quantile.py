import pandas as pd
print(pd.__version__)

df = pd.DataFrame({'cat': ['A','A','A','A','A','B','B','B','B','B'],\
                    'sales': [10,20,30,40,50,1,2,3,4,5]})

#   cat  sales
# 0   A     10
# 1   A     20
# 2   A     30
# 3   A     40
# 4   A     50
# 5   B      1
# 6   B      2
# 7   B      3
# 8   B      4
# 9   B      5

a = df['sales'].quantile(q=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], interpolation='nearest')        #5
# df.groupby(['cat'])['sales'].quantile(q=0.50, interpolation='nearest'
# A    30
# B     3

# df['cat_q50'] = df.groupby(['cat'])['sales'].transform(lambda x: x.quantile(q=0.50, interpolation='nearest'))
# df['cat_q50_without_min'] = df.groupby(['cat'])['sales'].transform(lambda x: x[x > x.min()].quantile(q=0.50, interpolation='nearest'))

print(a)