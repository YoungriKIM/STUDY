x_train = 0.5
y_train = 0.8
# 0.5를 넣었을 때 0.8이 나온다는 뜻

# 이 부분 바꿔가며 찍어보기 -------------------------------------
weight = 0.5    # 0.3, 0.66, 0.1, 1, 10 등
lr = 0.02       # 0.1, 10, 100, 0.001, 0.002 등
epoch = 100     # 10, 10000, 200 등
# 이 부분 바꿔가며 찍어보기 -------------------------------------

for iteration in range(epoch):
    y_predict = x_train * weight    # +bias 는 이 뒤에 붙이면 된다.
    error = (y_predict - y_train) **2   # loss = cost = error

    print('Error: ' + str(error) + '\ty_predict: ' + str(y_predict))

    up_y_predict = x_train * (weight + lr)
    up_error = (y_train - up_y_predict) ** 2    # 두개를 뺀 거에서 제곱을 한거는 양수 만드려는 거고 거리를 재려는 것임: mse

    print("up_y_predict: ", up_y_predict, '\tup_error: ', up_error) # 내가 편하게 보려고 붙인 부분

    down_y_predict = x_train * (weight - lr)
    down_error = (y_train - down_y_predict) ** 2
    
    print("down_y_predict: ", down_y_predict, '\tdown_error: ', down_error, '\n{:02d}'.format(iteration+1),'-----------') # 내가 편하게 보려고 붙인 부분

    if(down_error <= up_error):
        wright = weight - lr
    if(down_error > up_error):
        weight = weight + lr        # 러닝레이트를 빼거나 더하며 왔다 갔다 하는 것을 표현함.
