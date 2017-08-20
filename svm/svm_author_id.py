#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

sys.path.append("../tools/")
from email_preprocess import preprocess
from mylib import fit_and_predict

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = SVC(kernel='rbf', C=10000)
accuracy = fit_and_predict(clf, features_train, features_test, labels_train, labels_test)
print("SVM {:.3}".format(accuracy))
pred = clf.predict(features_test)
print(sum(i > 0 for i in pred)) # select quantity of Chris's emails

features_train = features_train[:len(features_train) // 100]
labels_train = labels_train[:len(labels_train) // 100]
for C in [10**i for i in range(0, 7)]:
    clf = SVC(kernel='rbf', C=C)
    accuracy = fit_and_predict(clf, features_train, features_test, labels_train, labels_test)
    print("SVM C={} | {:.3}".format(C, accuracy))

