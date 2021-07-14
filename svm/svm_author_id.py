#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

features_train, labels_train, features_test, labels_test = makeTerrainData()
clf = SVC(kernel="linear")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

def submitAccuracy():
    return accuracy_score(pred, labels_test)

print accuracy_score(pred, labels_test)



clf = SVC(kernel="linear", gamma=1.0)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

%matplotlib inline
prettyPicture(clf, features_test, labels_test)






clf = SVC(kernel="rbf", C=10**5)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

%matplotlib inline
prettyPicture(clf, features_test, labels_test)


clf = SVC(kernel="rbf", gamma=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

%matplotlib inline
prettyPicture(clf, features_test, labels_test)

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels








