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
### your code goes here ###
from sklearn.naive_bayes import GuassianNB

#creating classifier
clf = GuassianNB()

#training the data in to the classsifier and also finding how much time it tool to run
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"

#predicting and finding how much time it took to run
t0 = time()
pred = clf.pred(features_test)
print "predicting time:", round(time()-t0, 3), "s"

#findind the accuracy using accuracy_score
from sklearn.metrics import accuracy_score
acccuracy = accuracy_score(labels_train,pred)
print ("Accuracy obtained: %0.4F %%" % (accuracy*100))

#########################################################
