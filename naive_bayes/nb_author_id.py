#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
from sklearn.naive_bayes import GaussianNB


sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
t2 = time()
features_train, features_test, labels_train, labels_test = preprocess()
print("preprocess time:", round(time()-t2, 3), "s")

# Train a Gaussian Naive Bayes Classifier

clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
accuracy = clf.score(features_test, labels_test)
print("prediction time:", round(time()-t1, 3), "s")
print('Accuracy:', accuracy)


# Pre process  : 19.666 secs
# Training time:  0.713, 0.731, 0.730
# Predict time :  0.190, 0.192, 0.189
