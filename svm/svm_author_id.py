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
from sklearn.metrics import accuracy_score
from sklearn import svm


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

clf = svm.SVC(kernel="rbf", C = 10000)
t0 = time()
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
clf.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"
# Training Time: 213 seconds
# Training Time w/ subset of data: 0.117 seconds
# Training Time w/ rbf kernel: 0.164 seconds
# Training Time w/ rbf kernel & C=10: 0.127 seconds
# Training Time w/ rbf kernel & C=100: 0.138 seconds
# Training Time w/ rbf kernel & C=1000: 0.127 seconds
# Training Time w/ rbf kernel & C=10000: 0.129 seconds
# Training Time (full) rbf kernel & C=10000: 137.442 seconds
t1 = time()
prediction = clf.predict(features_test)
print "Prediction time:", round(time()-t1, 3), "s"
# Prediction Time: 23 second
# Prediction Time w/ subset of data: 0.88 seconds
# Prediction Time w/ rbf kernel: 1.475 seconds
# Prediction Time w/ rbf kernel & C=10: 1.465 seconds
# Prediction Time w/ rbf kernel & C=100: 1.338 seconds
# Prediction Time w/ rbf kernel & C=1000: 1.279 seconds
# Prediction Time w/ rbf kernel & C=10000: 1.159 seconds
# Prediction Time (full) rbf kernel & C=10000: 14.656 seconds
print accuracy_score(prediction, labels_test)
# Accuracy: 0.984072810011
# Accuracy w/ subset: 0.884527872582
# Accuracy w/ rbf kernel: 0.616040955631
# Accuracy w/ rbf kernel & C=10 or 100: 0.616040955631
# Accuracy w/ rbf kernel & C=1000: 0.821387940842
# Accuracy w/ rbf kernel & C=10000: 0.892491467577
# Accuracy (full) rbf kernel & C=10000: 0.990898748578

#Prediction for feature_test[10][26][50] = [1][0][1]
#print (prediction[10])
#print (prediction[26])
#print (prediction[50])

chris = []
# Get number of predicted emails written by Chris.  Ans: 877
for i in prediction:
    if i == 1:
        chris.append(i)

print len(chris)

#########################################################


