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

import numpy


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
from sklearn import svm

vectorOfOnes = numpy.ones((len(labels_test),), dtype=numpy.int)


# Training
print "Training"
t0 = time()
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
clf = svm.SVC(C=10000.0, kernel='rbf')
clf.kernel
clf.fit(features_train, labels_train)
print "time to train:", round(time() - t0, 3), "s"

# Predicting
print "Predicting"
t0 = time()
labels_pred = clf.predict(features_test)
print "time to predict:", round(time() - t0, 3), "s"

accuracy = 100 * float((labels_test == labels_pred).sum()) / float(features_test.shape[0])

print("Accuracy = %f" % accuracy)

print("Number of mislabeled points out of a total %d points : %d" % (
features_test.shape[0], (labels_test != labels_pred).sum()))

print "[10] Real: ", labels_test[10], " Predicted: ", labels_pred[10]
print "[26] Real: ", labels_test[26], " Predicted: ", labels_pred[26]
print "[50] Real: ", labels_test[50], " Predicted: ", labels_pred[50]

print("Number of events belonging to Chris (1) out of a total %d points : %d" % (
features_test.shape[0], (labels_pred == vectorOfOnes).sum()))

#########################################################


