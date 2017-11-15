import numpy as np
import pandas as pd
import random

# generate sample data
random.seed(2016)
sample_size = 50
sample = pd.Series(random.sample(range(-10000, 10000), sample_size))
x = sample / 10000
y = x**2

print x.head(10)

print x.describe()

count = 0
dataSet = [([x.ix[count]], [y.ix[count]])]
count = 1
while (count < sample_size):
    print "Working on data item: ", count
    dataSet = (dataSet + [([x.ix[count, 0]],  [y.ix[count]])])
    count = count + 1

# train
import neuralpy

fit = neuralpy.Network(1,3,7,1)
epochs = 100 
learning_rate = 1
print "fitting model right now" 
fit.train(dataSet, epochs, learning_rate) 

#predict
count = 0 
pred=[] 
while (count < sample_size): 
    out=fit.forward(x[count]) 
    print ("Obs: ",count+1,
            " y = ", round(y[count],4), 
            " prediction = ",
            round(pd.Series(out),4))
    pred.append(out)
    count = count + 1

out=fit.forward(0.3) 
print 'obs: ', 0.3, 'y= 0.09', 'prediction = ', round(pd.Series(out),4)

out=fit.forward(1) 
print 'obs: ', 1, 'y= 1', 'prediction = ', round(pd.Series(out),4)
