#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

import remove_outlier
from copy import copy
from select_k_best_features import get_k_best

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### features_list = ['poi','salary'] # You will need to use more 

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### List of persons
persons = data_dict.keys()

target_label = 'poi'

features_list = [
    'poi',
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]

# get K-best features
num_features = 3 # 10 for logistic regression, 8 for k-means clustering
best_features = get_k_best(data_dict, features_list, num_features)

feature_list = [target_label] + best_features

# features_list = ['poi','salary','exercised_stock_options', 'bonus']

### Task 2: Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
remove_outlier.remove_outlier(data_dict, outliers)

### Task 3: Create new feature(s)
# add two new features
# poi_ratio.add_poi_ratio(data_dict, features_list)

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### from sklearn.naive_bayes import GaussianNB
### clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

### from sklearn.tree import DecisionTreeClassifier
### clf = DecisionTreeClassifier(min_samples_leaf=2)

###FINAL CHOSEN ALGORITHM AND PARAMETERS - Better Recall, more True Positives
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(algorithm='auto', leaf_size=50, metric='minkowski',
           metric_params=None, n_neighbors=4, p=2, weights='distance')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, data_dict, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

### dump_classifier_and_data(clf, data_dict, features_list)