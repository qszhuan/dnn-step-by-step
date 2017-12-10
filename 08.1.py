import numpy as np 
loc= ".\PimaIndians_CleanData.txt" 
data = np.loadtxt(loc,skiprows=0) 
y=data[:,0] 
x=data[:,1:5]


# generate training and test set.
import random 
from sklearn.cross_validation import train_test_split
np.random.seed(2016) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# specify the model
from sknn.mlp import Classifier, Layer
from datetime import datetime

beg = datetime.now()

fit2 = Classifier(layers=[
    Layer("Tanh", units=45),
    Layer("Tanh", units=18),
    Layer("Tanh", units=18),
    Layer("Softmax")], 
    learning_rate=0.8,
    random_state=2016,
    valid_size=0.25,
    learning_momentum=0.30,
    batch_size=35,
    n_iter=100)

print "fitting model right now"
fit2.fit(x_train, y_train)
end = datetime.now()
print 'Time spent:', end-beg

from sklearn.metrics import confusion_matrix
pred2_train = fit2.predict(x_train)
confu2 = confusion_matrix(y_train , pred2_train)
print confu2

score2 = fit2.score (x_train , y_train )
print score2


# unbanlanced data

w_train = x_train[: ,0]
w_train[y_train == 0] = 1
w_train[y_train == 1] = 1.10


fit3 = Classifier(layers=[
    Layer("Tanh", units=45),
    Layer("Tanh", units=18),
    Layer("Tanh", units=18),
    Layer("Softmax")], 
    learning_rate=0.8,
    random_state=2016,
    valid_size=0.25,
    learning_momentum=0.30,
    batch_size=35,
    n_iter=100)

print "fitting model right now"
fit3.fit(x_train, y_train, w_train)
pred3_train = fit3.predict(x_train)
score3 = fit3.score(x_train, y_train)
print score3

confu3 = confusion_matrix(y_train, pred3_train)
print confu3