#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
from sklearn.svm import SVC

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


for j in [10000.0]:
    t= SVC(kernel="rbf",C=j)

    t= t.fit(features_train[:10000],labels_train[:10000])

    ans = t.predict(features_test)
    print 1
    deno = len(ans)
    count=0.0
    for i in range(len(features_test)):
        if ans[i] == labels_test[i]:
            count+=1

    print(count/deno , j)
#########################################################
### your code goes here ###

#########################################################


