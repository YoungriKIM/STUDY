import numpy as np

# a = np.array(range(0,10))
# size = 5

# print(a)
# print(len(a))

# def split_x(seq, size):
#     aaa = []
#     for i in range(len(seq) - size +1):
#         subset = seq[i : (i+size)]
#         aaa.append(subset)
#     print(type(aaa))
#     return np.array(aaa)

# dataset = split_x(a, size)

# print(dataset)
#-------------------------------

# 다:1 p.206
# dataset = np.array(range(1, 11))
# # print(dataset)

# def split_xy1(dataset, time_steps):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         end_number = i + time_steps
#         if end_number > len(dataset) -1:
#             break
#         tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)

# x, y = split_xy1(dataset, 3)
# print(x, '\n', y)
# print(x.shape)
# print(y.shape)

#-----------------------------
# 다:다 p.210
# dataset = np.array(range(1, 11))

# def split_xy2(dataset, time_steps, y_column):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps
#         y_end_number = x_end_number + y_column
#         # if end_number > len(dataset) -1:              # 위의 것에서 삭제
#         #     break
#         if y_end_number > len(dataset):
#             break
#         tmp_x = dataset[i : x_end_number]
#         tmp_y = dataset[x_end_number : y_end_number]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)

# time_steps = 4
# y_column = 2

# x, y = split_xy2(dataset, 4, 2)
# print(x, '\n', y)
# print(x.shape)
# print(y.shape)

#-----------------------------
# # 다입력, 다:1

# dataset = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],\
#      [21,22,23,24,25,26,27,28,29,30]])

# dataset = np.transpose(dataset)
# print(dataset.shape)

# def split_xy3(dataset, time_steps, y_column):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps
#         y_end_number = x_end_number + y_column -1

#         if y_end_number > len(dataset):
#             break
#         tmp_x = dataset[i:x_end_number, :-1]
#         tmp_y = dataset[x_end_number-1:y_end_number, -1]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)

# x, y = split_xy3(dataset, 3, 1)
# print(x, '\n', y)
# print(x.shape)
# print(y.shape)

#-----------------------------
# 다입력, 다:다

# dataset = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20], \
#     [21,22,23,24,25,26,27,28,29,30]])
# dataset = np.transpose(dataset)

# def split_xy4(dataset, time_steps, y_column):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps
#         y_end_number = x_end_number + y_column -1

#         if y_end_number > len(dataset):
#             break
#         tmp_x = dataset[i:x_end_number, :-1]
#         tmp_y = dataset[x_end_number-1:y_end_number, -1]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)

# x, y = split_xy4(dataset, 3, 2)
# print(x, '\n', y)
# print(x.shape) 
# print(y.shape)  

#-----------------------------
# 다입력, 다:다 두번째

# dataset = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20], \
#     [21,22,23,24,25,26,27,28,29,30]])
# dataset = np.transpose(dataset)

# def split_xy5(dataset, time_steps, y_column):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps
#         y_end_number = x_end_number + y_column

#         if y_end_number > len(dataset):
#             break
#         tmp_x = dataset[i:x_end_number, :]
#         tmp_y = dataset[x_end_number:y_end_number, :]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)

# x, y = split_xy5(dataset, 3, 1)
# print(x, '\n', y)
# print(x.shape)
# print(y.shape)
# (7, 3, 3)
# (7, 1, 3)

#-----------------------------
# 다입력, 다:다 세번째

dataset = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20], \
    [21,22,23,24,25,26,27,28,29,30],[31,32,33,34,35,36,37,38,39,40]])
dataset = np.transpose(dataset)
print(dataset.shape)        #(10, 4)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :] 
        tmp_y = dataset[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(dataset, 3, 2)
print(x, '\n', y)
print(x.shape)
print(y.shape)
