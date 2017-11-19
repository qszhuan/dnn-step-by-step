from sklearn import datasets
boston = datasets.load_boston()
x, y = boston.data, boston.target

from sklearn import preprocessing

x_MinMax = preprocessing.MinMaxScaler ()
y_MinMax = preprocessing.MinMaxScaler ()

import numpy as np
y = np.array(y).reshape((len(y), 1))
x = x_MinMax.fit_transform(x)
y = y_MinMax.fit_transform(y)
x.mean(axis =0)

print x_MinMax.scale_
print y_MinMax.scale_


import random
from sklearn.cross_validation import train_test_split

np.random.seed(2016)
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size = 0.2)

print len(x_train) 
print y_train
print y_test

from sknn.mlp import Regressor, Layer
fit1 = Regressor(layers=[
    Layer("Sigmoid", units=6),
    Layer("Sigmoid", units=14),
    Layer("Linear")],
    learning_rate=0.02,
    random_state=2016,
    n_iter=10)

print "fitting model right now", fit1.fit(x_train,y_train)


pred1_train = fit1.predict(x_train)
from sklearn.metrics import mean_squared_error
mse_1 = mean_squared_error(pred1_train, y_train)

print "Train ERROR = ", mse_1 
