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



t0 = time()
#########################################################
### your code goes here ###
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train , labels_train)
print("time taken to train is " , round((time()- t0), 3) , "seconds")
print(len("length of features of 0 ",features_train[0]))

results = clf.predict(features_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(results , labels_test)

print("accuracy score is " , acc)




#########################################################


