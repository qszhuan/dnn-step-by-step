from sklearn.datasets import load_digits

digits = load_digits()

print(digits.data.shape)

#print digits.DESCR

import pylab as pl

pl.imshow(digits.images[0], cmap=pl.cm.gray_r, interpolation='nearest')

pl.show()

print(digits.target[0])
