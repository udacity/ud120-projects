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
from sklearn import svm

def print_elapsed(start, end):
    print('elapsed: {}'.format(end - start))


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
clf = svm.SVC(kernel='rbf', C=10000)

# train
print('training...')
start = time()
clf.fit(features_train, labels_train)
print_elapsed(start, time())
accuracy_score = clf.score(features_test, labels_test)
print('accuracy: {}'.format(accuracy_score))

# make predictions on test data
predictions = clf.predict(features_test)
print(predictions)
