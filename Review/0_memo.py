import numpy as np

dataset = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9],[6,7,8,9,10]])
print(dataset.shape)    #(6, 5)

def split_xy(dataset, x_row, y_row):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + x_row
        y_end_number = x_end_number + y_row

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy(dataset, 3, 2)

print(x)
print(y)
print(x.shape) 
print(y.shape)