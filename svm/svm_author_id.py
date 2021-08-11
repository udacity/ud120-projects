#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("C:/Users/Austin/udacity/ud120-projects/svm")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]


#########################################################
### your code goes here ###
from sklearn.svm import SVC

clf = SVC(kernel = 'rbf', C = 10000.0)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
##checking for Chris prediction    
count_ = 0
for j in pred:
    if j == 1:
        count_ += 1
    else:
        count_ == 0

print(count_)
    
print("The accuracy score is {}".format(submitAccuracy()))
print(pred[10], pred[26], pred[50])

t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

#########################################################