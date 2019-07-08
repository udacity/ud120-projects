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

# Note - clf.predict() returns a list of continuous variables (between 0 and 1) while the labels_test array is binary (0 and 1).
# You need to use np.around(pred) to convert the predictions to closest int to be able to use 
# scikit-learn's classification accuracy_score method.


#########################################################
### your code goes here ###

#########################################################


