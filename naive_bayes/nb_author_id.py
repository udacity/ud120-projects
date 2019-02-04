#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
from sklearn.naive_bayes import GaussianNB  
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf=GaussianNB()
clf.fit(features_train,labels_train)
t0=time()
print('trainingtime',round(time()-t0,3),'s')

acc=clf.score(features_test, labels_test)
t0=time()
print('trainingtime',round(time()-t0,3),'s')

print(acc)



#########################################################
### your code goes here ###



#########################################################


