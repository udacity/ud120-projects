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

t0 = time()
gnb = gnb.fit(features_train, labels_train)
print "training time fit():", round(time()-t0, 3), "s"

t0 = time()
results_test = gnb.predict(features_test)
print "training time predict():", round(time()-t0, 3), "s"

accuracy = float(sum([1 if test == result else 0 for test, result in zip(results_test, labels_test)])) / len(labels_test)

print accuracy



#########################################################
