# https://wikidocs.net/52460 참고

import torch

# 1차원 -------------------------------------------------------
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])

print(t)                # tensor([0., 1., 2., 3., 4., 5., 6.])

print(t.dim())          # 차원: 1
print(t.shape)          # torch.Size([7])
print(t.size())         # torch.Size([7])

print(t[0], t[1], t[-1])    # tensor(0.) tensor(1.) tensor(6.)
print(t[2:5], t[4:-1])      # tensor([2., 3., 4.]) tensor([4., 5.])
print(t[:2], t[3:])         # tensor([0., 1.]) tensor([3., 4., 5., 6.])

# 2차원  -------------------------------------------------------
t = torch.FloatTensor([[1.,2.,3.],
                      [4.,5.,6.],
                      [7.,8.,9.],
                      [10.,11.,12.]])

print(t)

print('ndim of t: ', t.ndim)            # ndim of t:  2
print('dim of t: ', t.dim())            # dim of t2:  2
print('shape of t: ', t.shape)          # shape of t:  torch.Size([4, 3])
print('size of t: ', t.size())          # size of t:  torch.Size([4, 3])

print(t[:, 1])                          # tensor([ 2.,  5.,  8., 11.])
print(t[:, 1].shape)                    # torch.Size([4])

print(t[:, :-1])                        # tensor([[ 1.,  2.],
                                                # [ 4.,  5.],
                                                # [ 7.,  8.],
                                                # [10., 11.]])
print(t[:, :-1].shape)                  # torch.Size([4, 2])

# 브로드캐스팅  -------------------------------------------------------
# 크기가 다른 행렬 또는 텐서에 대해서 자동으로 크기를 맞춰주는 기능

m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1+m2)        #tensor([[5., 5.]])

