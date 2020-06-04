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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
import sklearn.svm import SVC
clf = SVC() #Default params here
clf.fit(features_train, features_test)
yhat = clf.predict(labels_train)

from sklearn.metrics import accuracy_score
acc = accuracy_score(yhat, labels_test)
print(acc)

#########################################################


