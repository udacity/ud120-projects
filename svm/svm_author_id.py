#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy as np
from time import time
#sys.path.append("../tools/")
from email_preprocess import preprocess

from tabulate import tabulate


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[0:len(features_train)/100]
labels_train = labels_train[0:len(labels_train)/100]

from sklearn.svm import SVC

c = 10000
t0 = time()
clf = SVC(kernel= 'rbf', C = c)

clf.fit(features_train,labels_train)

t1 = time()-t0

print "score of the SVM for C : %d is : %r" %(c, clf.score(features_test,labels_test))


print "time taken to process entire dataset is %r seconds" %t1
#########################################################
### your code goes here ###


predictions = clf.predict(features_test)

print len(predictions[predictions==1])

#print "10th is %d, 26th is %d and 50th is %d " %(predictions[10], predictions[26], predictions[50])
#########################################################


