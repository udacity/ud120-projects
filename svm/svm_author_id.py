#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
import numpy as np
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
def all_training_data(features_train, features_test, labels_train, labels_test):
    t0 = time()
    clf = SVC(kernel="linear")
    clf.fit(features_train, labels_train)
    print("Training Time: {time}".format(time=round(time() - t0, 3)))
    # Training Time: 187.015

    t0 = time()
    prediction = clf.predict(features_test)
    print("Prediction Time: {time}".format(time=round(time() - t0, 3)))
    # Prediction Time: 19.888

    acc = accuracy_score(labels_test, prediction)
    print(acc)
    #Accuracy: 0.9840728100113766

def one_percent_training_data(features_train, features_test, labels_train, labels_test):
    t0 = time()
    clf = SVC(kernel="linear")
    features_train = features_train[:len(features_train)//100]
    labels_train = labels_train[:len(labels_train)//100]

    clf.fit(features_train, labels_train)
    print("Training Time: {time}".format(time=round(time() - t0, 3)))
    #

    t0 = time()
    prediction = clf.predict(features_test)
    print("Prediction Time: {time}".format(time=round(time() - t0, 3)))
    #

    acc = accuracy_score(labels_test, prediction)
    print(acc)
    #Accuracy: 0.8845278725824801

def one_percent_training_data_rbf(features_train, features_test, labels_train, labels_test):
    t0 = time()
    clf = SVC(kernel="rbf")
    features_train = features_train[:len(features_train)//100]
    labels_train = labels_train[:len(labels_train)//100]

    clf.fit(features_train, labels_train)
    print("Training Time: {time}".format(time=round(time() - t0, 3)))
    # Training Time: 0.116

    t0 = time()
    prediction = clf.predict(features_test)
    print("Prediction Time: {time}".format(time=round(time() - t0, 3)))
    # Prediction Time: 1.213

    acc = accuracy_score(labels_test, prediction)
    print(acc)
    #Accuracy: 0.616

def one_percent_training_data_diff_c_levels(features_train, features_test, labels_train, labels_test):
    t0 = time()
    clf = SVC(kernel="rbf", C=10000)
    features_train = features_train[:len(features_train)//100]
    labels_train = labels_train[:len(labels_train)//100]

    clf.fit(features_train, labels_train)
    print("Training Time: {time}".format(time=round(time() - t0, 3)))

    t0 = time()
    prediction = clf.predict(features_test)
    print("Prediction Time: {time}".format(time=round(time() - t0, 3)))

    acc = accuracy_score(labels_test, prediction)
    print(acc)
    #C:10 -Accuracy: 0.61604
    #C:100 -Accuracy: 0.61604095
    #C:1 000 -Accuracy: 0.8213879
    #C:10 000 -Accuracy: 0.89249146757

def all_data_large_c_levels(features_train, features_test, labels_train, labels_test):
    t0 = time()
    clf = SVC(kernel="rbf", C=10000)

    clf.fit(features_train, labels_train)
    print("Training Time: {time}".format(time=round(time() - t0, 3)))

    t0 = time()
    prediction = clf.predict(features_test)
    print("Prediction Time: {time}".format(time=round(time() - t0, 3)))

    acc = accuracy_score(labels_test, prediction)
    print(acc)
    #Accuracy: 0.9908

def less_data_predictions(features_train, features_test, labels_train, labels_test):
    t0 = time()
    clf = SVC(kernel="rbf", C=10000)
    features_train = features_train[:len(features_train)//100]
    labels_train = labels_train[:len(labels_train)//100]

    clf.fit(features_train, labels_train)
    print("Training Time: {time}".format(time=round(time() - t0, 3)))

    t0 = time()
    predictions = clf.predict(features_test)
    print("Prediction Time: {time}".format(time=round(time() - t0, 3)))

    # acc = accuracy_score(labels_test, prediction)
    print(predictions[10])
    # 1
    print(predictions[26])
    # 0
    print(predictions[50])
    # 1

def total_chris_predictions(features_train, features_test, labels_train, labels_test):
    t0 = time()
    clf = SVC(kernel="rbf", C=10000)

    clf.fit(features_train, labels_train)
    features_train = features_train[:len(features_train)//100]
    labels_train = labels_train[:len(labels_train)//100]
    print("Training Time: {time}".format(time=round(time() - t0, 3)))

    t0 = time()
    predictions = clf.predict(features_test)
    print("Prediction Time: {time}".format(time=round(time() - t0, 3)))

    print(np.count_nonzero(predictions == 1))
    # Total Chris Predictions: 877

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC # Support vector classifier
from time import time

# Linear kernel results in a straight line decisions boundry
# Other Kernals:
## Poly
## Rbf - squigly decision boundry
## Sigmoid
## Precomputed
## Callable
# ---------------------------------------------------------------------
# Other params supported by SVC
## C - More training points correct the higher the value
## Gamma - Higher the value the more the algorithm will try match

#all_training_data(features_train, features_test, labels_train, labels_test)
#one_percent_training_data(features_train, features_test, labels_train, labels_test)
#one_percent_training_data_rbf(features_train, features_test, labels_train, labels_test)
#one_percent_training_data_diff_c_levels(features_train, features_test, labels_train, labels_test)
#all_data_large_c_levels(features_train, features_test, labels_train, labels_test)
#less_data_predictions(features_train, features_test, labels_train, labels_test)
total_chris_predictions(features_train, features_test, labels_train, labels_test)

# SVM training and predictions are much slower then Naive Bayes
#########################################################


