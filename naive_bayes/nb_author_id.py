#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from tools.email_preprocess import preprocess


def predict_author_id():
    # features_train and features_test are the features for the training
    # and testing datasets, respectively
    # labels_train and labels_test are the corresponding item labels
    features_train, features_test, labels_train, labels_test = preprocess()
    gnb = GaussianNB()

    training_start = time()
    gnb.fit(features_train, labels_train)
    print("training time: {}s".format(round(time()-training_start, 3)))

    prediction_start = time()
    prediction = gnb.predict(features_test)
    print("prediction time: {}s".format(round(time()-prediction_start, 3)))

    print(accuracy_score(labels_test, prediction, normalize=True))


if __name__ == "__main__":
    predict_author_id()
