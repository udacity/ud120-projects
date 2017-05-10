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

from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()
clf.fit(features_train,labels_train)
predictions=clf.predict(features_test)

total=len(labels_test)
errors=0

for i in range(total):
    if(predictions[i]!=labels_test[i]):
        errors+=1
correct=float(total-errors)
accuracy=correct/total


total=float(total)

print clf.score(features_test, labels_test)




#########################################################


