#!/usr/bin/python3

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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time
from collections import Counter
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# features_train = features_train[:len(features_train)//100]
# labels_train = labels_train[:len(labels_train)//100]

### make sure you use // when dividing for integer division

#clf =  SVC(kernel = 'linear') 		#for linear kernel
clf = SVC(kernel='rbf', C=10000)	#for rbf kernel
t0 = time()
clf.fit(features_train, labels_train)
print('training time', time()-t0)

t1=time()
pred = clf.predict(features_test)
print('predicting time', time()-t1)
accuracy = accuracy_score(pred, labels_test)

print(accuracy)
print(pred[10], pred[26], pred[50])

counter = Counter(pred)
print("'Chris 1's: ", counter[1])
#########################################################
### your code goes here ###

#########################################################


