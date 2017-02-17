#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import math
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def chrisOrSara(pred_array,position):
    res = pred_array[position]
    
    if res==0:
        person="Sara"
    elif res==1:
        person="Chris"
    print ("Person at position ",position," is ",person)
    
def chris_total(predictions):
    total=0
    for i in pred:
        if i==1 : 
            total+=1
    return total

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
clf = SVC(kernel='rbf', gamma='auto', C=10000)

# to speed up computation
#features_train = features_train[:math.ceil(len(features_train)/100)] 
#labels_train = labels_train[:math.ceil(len(labels_train)/100)] 

t0 = time()
clf.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time() 
pred = clf.predict(features_test)
print ("predict time:", round(time()-t0, 3), "s")

print ("Accuracy score:", accuracy_score(pred,labels_test))

chrisOrSara(pred,10)
chrisOrSara(pred,26)
chrisOrSara(pred,50)
print ("Chris total ", chris_total(pred))
#########################################################


