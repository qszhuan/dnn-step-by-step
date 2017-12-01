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

fit1 = Classifier(layers=[
    Layer("Sigmoid", units=45),
    Layer("Sigmoid", units=18),
    Layer("Sigmoid", units=18),
    Layer("Softmax")], 
    learning_rate=0.25,
    random_state=2016,
    valid_size=0.25,
    learning_momentum=0.30,
    n_iter=100)

print "fitting model right now"
fit1.fit(x_train, y_train)

prob = fit1.predict_proba(x_train)

# confusion matrix
from sklearn.metrics import confusion_matrix
pred1_train = fit1.predict(x_train) 

confu1 = confusion_matrix(y_train, pred1_train)

print confu1