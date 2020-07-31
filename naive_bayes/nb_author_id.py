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
# create classifer
clf = GaussianNB()

# note time
t0 = time()

# fit the classifier on  training features and labels
clf.fit(features_train, labels_train)
print("Training time", time()-t0, "s")

# note time
t1=time()

# predict labels for the test features
pred = clf.predict(features_test)
print("Predicting time", time()-t1, "s")

# calculate accuracy
accuracy = accuracy_score(pred, labels_test)

# return the accuracy
# return(accuracy)

print(accuracy)
# return the accuracy

#########################################################


