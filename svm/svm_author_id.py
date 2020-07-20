#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time

from tools.email_preprocess import preprocess

from sklearn import svm
from sklearn.metrics import accuracy_score


def predict_author_id():
    # features_train and features_test are the features for the training
    # and testing datasets, respectively
    # labels_train and labels_test are the corresponding item labels
    features_train, features_test, labels_train, labels_test = preprocess()

    # features_train = features_train[:len(features_train)/100]
    # labels_train = labels_train[:len(labels_train)/100]

    clf = svm.SVC(C=10000, kernel='rbf')
    clf.fit(features_train, labels_train)

    prediction = clf.predict(features_test)

    print("accuracy: {}".format(accuracy_score(labels_test, prediction, normalize=True)))
    print("prediction at position 10: {}".format(prediction[10]))
    print("prediction at position 26: {}".format(prediction[26]))
    print("prediction at position 50: {}".format(prediction[50]))
    print("no. of results predicted to be Chris (1): {}".format(sum(map(lambda x: x == 1, prediction))))


if __name__ == "__main__":
    predict_author_id()
