#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

# clf = LinearSVC()
clf = SVC(kernel='rbf', C=10000.0)

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

accuracy=accuracy_score(labels_test, pred)

count = 0
for x in pred:
    count += x

print sum(pred)
print count
#########################################################
