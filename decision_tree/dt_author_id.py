#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("./tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create a Decision Tree Classifier (DT) object
t0 = time()
clf = DecisionTreeClassifier(random_state=0, min_samples_split=40)
clf.fit(features_train, labels_train)        
print("Training Time:", round(time()-t0, 3), "s")

#########################################################

pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)

acc = accuracy_score(pred, labels_test)

print("Accuracy:", round(accuracy,3))
print("Metrics Accuracy:", round(acc, 3))

#########################################################

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Create a AdaBoost Classifier (AB) object
t0 = time()
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

acc = accuracy_score(clf.predict(features_test), labels_test)

print("Metrics Accuracy:", round(acc, 3))


#########################################################
from sklearn.neighbors import KNeighborsClassifier

# Create a KNeighbors Classifier (KNN) object
t0 = time()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

acc = accuracy_score(clf.predict(features_test), labels_test)
print("Metrics Accuracy:", round(acc, 3))


#########################################################

from sklearn.ensemble import RandomForestClassifier

# Create a RandomForest Classifier (RF) object
t0 = time()
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

acc = accuracy_score(clf.predict(features_test), labels_test)
print("Metrics Accuracy:", round(acc, 3))
