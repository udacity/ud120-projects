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

sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

## Training
t0 = time()
gnb.fit(features_train, labels_train)
print "time to train:", round(time() - t0, 3), "s"

## Predicting
t0 = time()
labels_pred = gnb.predict(features_test)
print "time to predict:", round(time() - t0, 3), "s"

accuracy = 100 * float((labels_test == labels_pred).sum()) / float(features_test.shape[0])

print("Accuracy = %f" % accuracy)

print("Number of mislabeled points out of a total %d points : %d" % (
features_test.shape[0], (labels_test != labels_pred).sum()))

#########################################################
