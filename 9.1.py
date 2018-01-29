import numpy as np
import random
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split


digits = load_digits()
np.random.seed(2016)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)


from sknn.mlp import Classifier, Layer

fit1 = Classifier(
    layers = [
        Layer("Tanh", units=21),
        Layer("Tanh", units=30),
        Layer("Sigmoid", units=37),
        Layer("Softmax")],
        valid_size=0.25,
        random_state = 2016,
        learning_rule = u'rmsprop',
        learning_momentum=0.005,
        learning_rate=0.30,
        dropout_rate=0.05,
        batch_size=34,
        n_iter=100
)

print " fitting model right now "
fit1.fit(x_train, y_train)
pred1_train = fit1.predict(x_train)
score1 = fit1.score (x_train, y_train)
print score1



fit2 = Classifier(
    layers = [
        Layer("Tanh", units=21),
        Layer("Tanh", units=30),
        Layer("Sigmoid", units=37),
        Layer("Softmax")],
        valid_size=0.25,
        random_state = 2016,
        learning_rule = u'adagrad',
        learning_momentum=0.005,
        learning_rate=0.30,
        dropout_rate=0.05,
        batch_size=34,
        n_iter=100
)

print " fitting model right now "
fit2.fit(x_train, y_train)
pred2_train = fit2.predict(x_train)
score2 = fit2.score (x_train, y_train)
print score2



fit3 = Classifier(
    layers = [
        Layer("Tanh", units=21),
        Layer("Tanh", units=30),
        Layer("Sigmoid", units=37),
        Layer("Softmax")],
        valid_size=0.25,
        random_state = 2016,
        learning_rule = u'nesterov',
        learning_momentum=0.005,
        learning_rate=0.30,
        dropout_rate=0.05,
        batch_size=34,
        n_iter=100
)

print " fitting model right now "
fit3.fit(x_train, y_train)
pred3_train = fit3.predict(x_train)
score3= fit3.score (x_train, y_train)
print score3


fit4 = Classifier(
    layers = [
        Layer("Tanh", units=21),
        Layer("Tanh", units=30),
        Layer("Sigmoid", units=37),
        Layer("Softmax")],
        valid_size=0.25,
        random_state = 2016,
        learning_rule = u'momentum',
        learning_momentum=0.005,
        learning_rate=0.30,
        dropout_rate=0.05,
        batch_size=34,
        n_iter=100
)

print " fitting model right now "
fit4.fit(x_train, y_train)
pred4_train = fit4.predict(x_train)
score4= fit4.score (x_train, y_train)
print score4



fit5 = Classifier(
    layers = [
        Layer("Tanh", units=21),
        Layer("Tanh", units=30),
        Layer("Sigmoid", units=37),
        Layer("Softmax")],
        valid_size=0.25,
        random_state = 2016,
        learning_rule = u'sgd',
        learning_momentum=0.005,
        learning_rate=0.30,
        dropout_rate=0.05,
        batch_size=34,
        n_iter=100
)

print " fitting model right now "
fit5.fit(x_train, y_train)
pred5_train = fit5.predict(x_train)
score5= fit5.score(x_train, y_train)
print score5



fit6 = Classifier(
    layers = [
        Layer("Tanh", units=21),
        Layer("Tanh", units=30),
        Layer("Sigmoid", units=37),
        Layer("Softmax")],
        valid_size=0.25,
        random_state = 2016,
        learning_rule = u'adadelta',
        learning_momentum=0.005,
        learning_rate=0.30,
        dropout_rate=0.05,
        batch_size=34,
        n_iter=100
)

print " fitting model right now "
fit6.fit(x_train, y_train)
pred6_train = fit6.predict(x_train)
score6= fit6.score(x_train, y_train)
print score6

score_test6 = fit6.score(x_test, y_test)
pred6_test = fit6.predict(x_test)

from sklearn.metrics import confusion_matrix
confu6_test = confusion_matrix(y_test, pred6_test)

print confu6_test
 