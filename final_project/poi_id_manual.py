#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### helper functions for data wrangling
from data_wrangling import *

### helper functions for Machine Learning
from data_learning import *

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
total_features_list = get_features(data_dict, target_label)

### Add valid counts and total counts for inspection
features_specs = count_values(data_dict, total_features_list)

### List of persons
persons_list = data_dict.keys()

### from manual inspection of the above list of persons I decided to choose
### 2 person TOTAL and THE TRAVEL AGENCY IN THE PARK as my outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']

### From here since we might need different number of features for different
### algorithms, we will perform the rest of steps separately for different
### algorithms.


################### Logistic Regression Classifier starts ###################

### Classifier
from sklearn.linear_model import LogisticRegression

### Store to my_dataset for easy export below.
my_dataset = data_dict

### get K-best features, feature scores, valid counts and total counts and
### create a feature list with first feature as label and rest of the best 
### features chosen above
num_features = 5
best_features, all_features_scores = get_k_best_features(my_dataset, 
                                                         total_features_list, 
                                                         num_features)
features_list = [target_label] + best_features

### Task 2: Remove outliers
my_dataset = remove_outlier(my_dataset, outliers)

### Task 3: Create new feature(s)

### Upon manual inspection of the features_list I observed that
### no feature related to their financial status such as stocks
my_lr_dataset, features_lr_list = add_features_totals(my_dataset, features_list)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Extract features and labels from dataset for local testing
data = featureFormat(my_lr_dataset, features_lr_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#### scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Parameter optimizer
total_score = 0.

### variable for best classifier
best_clf = None

### From my previous runs, I found these parameters after optimization for 6
### best features and 2 added features. Comment the below line to find the 
### optimized parameters again. Parameters might come out to be different 
### in different runs
best_clf = LogisticRegression(C=10**12, tol=10**-15, class_weight='auto')

if not best_clf:
  for i in range(0, 18):
    for j in range(0, 18):
      clf = LogisticRegression(C=10**i, tol=10**j, class_weight='auto')
      precision, recall = get_scores(clf, my_lr_dataset, features_lr_list)
      if (precision >= 0.3) and (recall >= 0.3) and (precision + recall > total_score):
        total_score = precision + recall
        best_clf = LogisticRegression(C=10**i, tol=10**-j, class_weight='auto')

if best_clf:
  best_lr_clf = best_clf
  test_classifier(best_lr_clf, my_lr_dataset, features_lr_list)
else:
  print 'Did not find parameters for best Logistic Regression \n'

################### Logistic Regression Classifier ends ###################

################### K Neighbors Classifier starts ###################

### Classifier
from sklearn.neighbors import KNeighborsClassifier

### Store to my_dataset for easy export below.
my_dataset = data_dict

### get K-best features, feature scores, valid counts and total counts and
### create a feature list with first feature as label and rest of the best 
### features chosen above
num_features = 16
best_features, all_features_scores = get_k_best_features(my_dataset, 
                                                         total_features_list, 
                                                         num_features)
features_list = [target_label] + best_features

### Task 2: Remove outliers
my_dataset = remove_outlier(my_dataset, outliers)

### Task 3: Create new feature(s)

### Upon manual inspection of the features_list I observed that
### no feature related to their financial status such as stocks
my_kn_dataset, features_kn_list = add_features_totals(my_dataset, features_list)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Extract features and labels from dataset for local testing
data = featureFormat(my_kn_dataset, features_kn_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#### scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Parameter optimizer
total_score = 0.

### variable for best classifier
best_clf = None

### From my previous runs, I found these parameters after optimization for 6
### best features and 2 added features. Comment the below line to find the 
### optimized parameters again. Parameters might come out to be different 
### in different runs
best_clf = KNeighborsClassifier(algorithm='auto', leaf_size=2**0, 
                                metric='minkowski', metric_params=None, 
                                n_neighbors=3, p=1, weights='distance')

if not best_clf:
  for i in range(5):
    for j in range(1, 6):
      for k in range(1, 6):
        clf = KNeighborsClassifier(algorithm='auto', leaf_size=2**i, 
                                   metric='minkowski', metric_params=None, 
                                   n_neighbors=j, p=k, weights='distance')
        precision, recall = get_scores(clf, my_kn_dataset, features_kn_list)
        if (precision >= 0.3) and (recall >= 0.3) and (precision + recall > total_score):
          total_score = precision + recall
          best_clf = KNeighborsClassifier(algorithm='auto', leaf_size=2**i, 
                                          metric='minkowski', metric_params=None, 
                                          n_neighbors=j, p=k, weights='distance')

if best_clf:
    best_kn_clf = best_clf
    test_classifier(best_kn_clf, my_kn_dataset, features_kn_list)
else:
    print 'Did not find parameters for best K Neighbors Regression \n'

################### K Neighbors Classifier ends ################### 

### After optimization both logistic regression and K Neighbors classifier give
### both precision and recall better than 0.3.

clf = best_kn_clf
features_list = features_kn_list
my_dataset = my_kn_dataset

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)