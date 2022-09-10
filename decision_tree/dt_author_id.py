#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
# sys.path.append("../tools/")
sys.path.append("..\\tools\\")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
# features_train, features_test, labels_train, labels_test = preprocess()
features_train, features_test, labels_train, labels_test = preprocess(words_file = "..\\tools\\word_data_unix.pkl", authors_file="..\\tools\\email_authors.pkl")

print("Number of training features =", len(features_train[0]))

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 40)
clf = clf.fit(features_train, labels_train)

acc = clf.score(features_test, labels_test)

print("accuracy with samples = 40 is:", acc)


