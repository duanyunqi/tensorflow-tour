import numpy as np

# A = np.arange(15).reshape(3, 5)
# print(A)
# np.save('A.npy', A)
# B = np.load('A.npy')
# print(B)

vgg16 = np.load('../vgg16.npy', encoding='latin1').item()
print(vgg16.keys())


