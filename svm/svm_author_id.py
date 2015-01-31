#!/usr/bin/python

"""
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors

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
from sklearn import svm
from sklearn.metrics import accuracy_score

linear_kernel_svm = svm.SVC(kernel='rbf', C=10000.)

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

t0 = time()
linear_kernel_svm.fit(features_train, labels_train)
print "training time with SVM's linear kernel", time() - t0

t1 = time()
pred = linear_kernel_svm.predict(features_test)
print "prediction time with SVM's linear kernel", time() - t1

acc = accuracy_score(labels_test, pred)
print acc

#########################################################

def time_with_power(power, people,times):
    results = nd.random.power(power, people)
    for i in range(times):
            results += nd.random.power(power, 1000)
    return results
