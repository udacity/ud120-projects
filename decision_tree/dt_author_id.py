#!/usr/bin/python

"""
    this is the code to accompany the Lesson 3 (decision tree) mini-project
    
    use an DT to identify emails from the Enron corpus by their authors
    
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

from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split = 40)

### fit the classifier on the training features and labels
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

### use the trained classifier to predict labels for the test features
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

### calculate and return the accuracy on the test data
accuracy = clf.score(features_test, labels_test)

print accuracy


#########################################################


