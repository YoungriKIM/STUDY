# ============================================
a = 72
b = a/255
print('b: \n', b)
print('b type: ', type(b))

# b: 
#  0.2823529411764706
# b type:  <class 'float'>
# ============================================
c = a/255.
print('c: \n', c)
print('c type: ', type(c))
# --------
# c:
#  0.2823529411764706
# c type:  <class 'float'>
# ============================================

d = int(c)
print(d)
print(type(d))