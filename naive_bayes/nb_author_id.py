#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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


#########################################################

def authour_Accuracy(features_train,labels_train,features_test,labels_test):
	from sklearn.naive_bayes import GaussianNB
	from sklearn.metrics import accuracy_score
	clf = GaussianNB()
	
	t0 = time()
	clf.fit(features_train,labels_train)
	t_fit=round(time()-t0,3)
	
	t1 = time()
	lables_predict = clf.predict(features_test)
	t_predict = round(time()-t1,3)
	#accuracy = clf.score(features_test,labels_test)
	accuracy = round(accuracy_score(labels_test,lables_predict),5)
	return accuracy,t_fit,t_predict 

print authour_Accuracy(features_train,labels_train,features_test,labels_test)