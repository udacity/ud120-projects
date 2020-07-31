#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
#clf = SVC(kernel='linear')
clf = SVC(kernel='rbf', C=10000)

t0 = time()
clf.fit(features_train, labels_train)
print( "training time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print( "Prediction time:", round(time()-t1, 3), "s")

print( "Accuracy score:", accuracy_score(labels_test, pred))

print( "Predictions for 10:", pred[10], print "Predictions for 26:", pred[26], print "Predictions for 50:", pred[50])

c = Counter(pred)
print "Number of predictions for Chris(1):", c[1]
#########################################################


