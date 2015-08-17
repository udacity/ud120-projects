#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from copy import copy
from evaluate import get_scores

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

### helper function for getting list of features
from get_features import features

### helper function to remove persons which are outliers
import remove_outlier

### helper function to select k best features
from select_k_best_features import get_k_best

### helper function to add new features
import add_features

### helper function to count valid and total values
import count_values

### helper functions for various pipelines
from model import *

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### features_list = ['poi','salary'] # You will need to use more 

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Target label to identify whether a person is poi or not
target_label = 'poi'

### Now get a comprehensive list of features from the data excluding 
### string features and first feature as poi which is the label
total_features_list = features(data_dict, target_label)

### get K-best features, feature scores, valid counts and total counts
num_features = 6
best_features, all_features_scores = get_k_best(data_dict, total_features_list, num_features)

### Add valid counts and total counts for inspection
features_specs = count_values.count_valid_values(data_dict, all_features_scores)

### create the final feature list with first feature as label and
### rest of the best feature chosen above
features_list = [target_label] + best_features

### Task 2: Remove outliers

### List of persons
persons_list = data_dict.keys()

### from manual inspection of the above list of persons I decided to remove
### 2 person TOTAL and THE TRAVEL AGENCY IN THE PARK from the data
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
remove_outlier.remove_outlier(data_dict, outliers)

### Task 3: Create new feature(s)

### Upon manual inspection of the features_list I observed that
### no feature related to their financial status such as stocks
data_dict, features_list = add_features.add_totals(data_dict, features_list)

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

################### Logistic Regression Classifier starts ###################

### Classifier
from sklearn.linear_model import LogisticRegression

### Parameter optimizer
total_score = 0.

### From my previous runs, I found these parameters after optimization for 6
### best features and 2 added features
best_clf = LogisticRegression(C=10**12, tol=10**-15, class_weight='auto')

### Uncomment the below line to run the parameter optimizer and find the optimized
### parameters again. Parameters might come out to be different in different runs
#best_clf = None

if not best_clf:
  for i in range(0, 28, 3):
    for j in range(0, 28, 3):
      clf = LogisticRegression(C=10**i, tol=10**-j, class_weight='auto')
      precision, recall = get_scores(clf, data_dict, features_list)
      if (precision >= 0.3) and (recall >= 0.3) and (precision + recall > total_score):
        total_score = precision + recall
        # print "i: {0}, j: {1}, precision: {2}, recall: {3}, total_score: {4}".format(i, j, precision, recall, total_score)
        best_clf = LogisticRegression(C=10**i, tol=10**-j, class_weight='auto')

if best_clf:
  best_lr_clf = best_clf
  #print 'here is the best Logistic Regression \n'
  #test_classifier(best_lr_clf, data_dict, features_list)
else:
  print 'Did not find parameters for best Logistic Regression \n'

################### Logistic Regression Classifier ends ###################

################### K Neighbors Classifier starts ###################

### Classifier
from sklearn.neighbors import KNeighborsClassifier

### Parameter optimizer
total_score = 0.

### From my previous runs, I found these parameters after optimization for 16
### best features and 2 added features
best_clf = KNeighborsClassifier(algorithm='auto', leaf_size=2**0, metric='minkowski',
                           metric_params=None, n_neighbors=3, p=1, weights='distance')

### Uncomment the below line to run the parameter optimizer and find the optimized
### parameters again. Parameters might come out to be different in different runs
#best_clf = None

if not best_clf:
  for i in range(5):
    for j in range(1, 6):
      for k in range(1, 6):
        clf = KNeighborsClassifier(algorithm='auto', leaf_size=2**i, metric='minkowski',
                                   metric_params=None, n_neighbors=j, p=k, weights='distance')
        precision, recall = get_scores(clf, data_dict, features_list)
        if (precision >= 0.3) and (recall >= 0.3) and (precision + recall > total_score):
          total_score = precision + recall
          print "i: {0}, j: {1}, k: {2}, precision: {3}, recall: {4}, total_score: {5}".format(i, j, k, precision, recall, total_score)
          best_clf = KNeighborsClassifier(algorithm='auto', leaf_size=2**i, metric='minkowski',
                                          metric_params=None, n_neighbors=j, p=k, weights='distance')

if best_clf:
    best_dt_clf = best_clf
    #print 'here is the best K Neighbors Regression \n'
    #test_classifier(best_dt_clf, data_dict, features_list)
else:
    print 'Did not find parameters for best K Neighbors Regression \n'

################### K Neighbors Classifier ends ###################

### from sklearn.naive_bayes import GaussianNB
### clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

### from sklearn.tree import DecisionTreeClassifier
### clf = DecisionTreeClassifier(min_samples_leaf=2)

###FINAL CHOSEN ALGORITHM AND PARAMETERS - Better Recall, more True Positives
#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(algorithm='auto', leaf_size=100, metric='minkowski',
#           metric_params=None, n_neighbors=3, p=2, weights='distance')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#test_classifier(clf, data_dict, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

### dump_classifier_and_data(clf, data_dict, features_list)