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
from sklearn.naive_bayes import GaussianNB

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

classifier = GaussianNB() #instantiate Naive Bayes classifier

t0 = time()
classifier.fit(features_train, labels_train) #train the classifier
print "training time:", round(time()-t0, 3), "s"

t1 = time()
classifier.predict(features_test) #make predictions on test data
print "prediction time:", round(time()-t1, 3), "s"

print "accuracy:", classifier.score(features_test, labels_test) #score classifier accuracy


#########################################################


