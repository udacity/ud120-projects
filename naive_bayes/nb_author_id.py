#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


def fit_and_predict(clf, features_train, features_test, labels_train, labels_test):
    t0 = time()
    clf.fit(features_train, labels_train)
    t1 = time()
    pred = clf.predict(features_test)
    t2 = time()
    accuracy = accuracy_score(labels_test, pred)
    t3 = time()
    print("{:05.3f} : {:05.3f} : {:05.3f} | {:05.3f}".format(t1-t0, t2-t1, t3-t2, t3-t0))
    return accuracy

print("{:5} : {:5} : {:5} | {:5}".format("fit", "pred", "acc", "total"))

clf = GaussianNB()
acc = fit_and_predict(clf, features_train, features_test, labels_train, labels_test)
print("Gaussian {0:.3}".format(acc))

clf = MultinomialNB()
acc = fit_and_predict(clf, features_train, features_test, labels_train, labels_test)
print("Multinomial {0:.3}".format(acc))

clf = BernoulliNB()
acc = fit_and_predict(clf, features_train, features_test, labels_train, labels_test)
print("Bernoulli {0:.3}".format(acc))
