#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print(f"# features: {len(features_train[0])}")

# clf = DecisionTreeClassifier(min_samples_split=40)
# clf.fit(features_train, labels_train)
# y_pred = clf.predict(features_test)
# acc = accuracy_score(labels_test, y_pred)

# print(f"Accuracy: {acc}")

#########################################################
