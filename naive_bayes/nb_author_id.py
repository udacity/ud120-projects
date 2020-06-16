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
base_path = "C:/Users/Trent.Park/Projects/udacity/ud120-projects"
sys.path.append(base_path + "/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

nb_model = GaussianNB()

t0 = time()
nb_model.fit(features_train, labels_train)
print "\ntraining time:", round(time() - t0, 3), "s\n"

t0  = time()
labels_pred = nb_model.predict(features_test)
print "simulation time:", round(time() - t0, 3), "s\n"

accuracy = accuracy_score(labels_test, labels_pred)

print "The model accuracy is: {0}\n".format(accuracy)

#########################################################


