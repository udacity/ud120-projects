#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=40)
clf.fit(features_train, labels_train)

labels_pred = clf.predict(features_test)

acc = accuracy_score(labels_test, labels_pred)

# Get the no. features by extracting the no. of columns in the training data
print "The number of features in the training data is {0}".format(len(features_train[0]))

print "\nThe Decision Tree accuracy is: {0}\n".format(acc)

#########################################################


