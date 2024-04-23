#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("./tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create a Support Vector Classifier (SVC) object with a linear kernel
clf = SVC(kernel='linear')

# Record the start time for training
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

# Record the start time for predicting
t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

# Calculate and print the accuracy of the model using the test data
accuracy = clf.score(features_test, labels_test)
print("Accuracy:", round(accuracy, 3))

# Import the accuracy_score function from scikit-learn's metrics module and calculate the accuracy of the model using the predicted values and actual labels
acc = accuracy_score(pred, labels_test)
print("Metrics Accuracy:", round(acc, 3))


#########################################################
# Training on smaller datasets
#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# Reduce the size of the features_train list to 1% of its original size
features_train = features_train[:int(len(features_train)/100)]

# Reduce the size of the labels_train list to 1% of its original size
labels_train = labels_train[:int(len(labels_train)/100)]


# Record the start time for training
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

# Record the start time for predicting
t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

# Calculate and print the accuracy of the model using the test data
accuracy = clf.score(features_test, labels_test)
print("Accuracy:", round(accuracy, 3))

# Import the accuracy_score function from scikit-learn's metrics module and calculate the accuracy of the model using the predicted values and actual labels
acc = accuracy_score(pred, labels_test)
print("Metrics Accuracy:", round(acc, 3))

#########################################################
# running the modle with RBF kernal on the small dataset
########################################################
# remember this modle running on the smaller data set of 1% of the oreginal data set
# Create a Support Vector Classifier (SVC) object with a RBF kernel
clf = SVC(kernel='rbf')

# Record the start time for training
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

# Record the start time for predicting
t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

# Calculate and print the accuracy of the model using the test data
accuracy = clf.score(features_test, labels_test)
print("Accuracy:", round(accuracy, 3))

# Import the accuracy_score function from scikit-learn's metrics module and calculate the accuracy of the model using the predicted values and actual labels
acc = accuracy_score(pred, labels_test)
print("Metrics Accuracy:", round(acc, 3))

#########################################################
# running the modle with different C values (10.0, 100., 1000., and 10000) kernal on the small dataset
########################################################
# remember this modle running on the smaller data set of 1% of the oreginal data set
# Create a Support Vector Classifier (SVC) object with a RBF kernel
c_values = [10.0, 100.0, 1000.0, 10000]

for c in c_values:
    clf = SVC(kernel='rbf', C=c)

    # Record the start time for training
    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training Time:", round(time()-t0, 3), "s")

    # Record the start time for predicting
    t0 = time()
    pred = clf.predict(features_test)
    print("Predicting Time:", round(time()-t0, 3), "s")

    # Calculate and print the accuracy of the model using the test data
    accuracy = clf.score(features_test, labels_test)
    print("Accuracy:", round(accuracy, 3))

    # Import the accuracy_score function from scikit-learn's metrics module and calculate the accuracy of the model using the predicted values and actual labels
    acc = accuracy_score(pred, labels_test)
    print("Metrics Accuracy:", round(acc, 3))
    
