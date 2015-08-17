#!/usr/bin/python

from copy import copy
import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

import enron
import evaluate

# features_list is a list of strings, each of which is a feature name
# first feature must be "poi", as this will be singled out as the label
target_label = 'poi'
email_features_list = [
    # 'email_address', # remit email address; informational label
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]
financial_features_list = [
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
features_list = [target_label] + financial_features_list + email_features_list

# load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# remove outliers
outlier_keys = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
enron.remove_keys(data_dict, outlier_keys)

# instantiate copies of dataset and features for grading purposes
my_dataset = copy(data_dict)
my_feature_list = copy(features_list)

# get K-best features
num_features = 10 # 10 for logistic regression, 8 for k-means clustering
best_features = enron.get_k_best(my_dataset, my_feature_list, num_features)
my_feature_list = [target_label] + best_features.keys()

# add two new features
enron.add_financial_aggregate(my_dataset, my_feature_list)
enron.add_poi_interaction(my_dataset, my_feature_list)

# print features
print "{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:])

# extract the features specified in features_list
data = featureFormat(my_dataset, my_feature_list)

# split into labels and features (this line assumes that the first
# feature in the array is the label, which is why "poi" must always
# be first in the features list
labels, features = targetFeatureSplit(data)

# scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# parameter optimization (not currently used)
from sklearn.grid_search import GridSearchCV

### Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
l_clf = None
###################################################
# brute-force parameter optimizer; uncomment to run
# TODO: use GridSearchCV
# k = 0
# best_combo = None
# max_exponent = 21
# for i in range(0, max_exponent, 3):
#     for j in range(0, max_exponent, 3):
#         print "i: {0}, j: {1}".format(i, j)
#         l_clf = LogisticRegression(C=10**i, tol=10**-j, class_weight='auto')
#         results = evaluate.evaluate_clf(l_clf, features, labels)
#         if sum(results) > k:
#             k = sum(results)
#             best_combo = (i, j)
# l_clf = LogisticRegression(C=10**i, tol=10**-j)
###################################################
if not l_clf:
    l_clf = LogisticRegression(C=10**18, tol=10**-21)

### K-means Clustering
from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.001)

### Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
a_clf = AdaBoostClassifier(algorithm='SAMME')

### Support Vector Machine Classifier
from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=1000)

### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()

### Stochastic Gradient Descent - Logistic Regression
from sklearn.linear_model import SGDClassifier
g_clf = SGDClassifier(loss='log')

### Selected Classifiers Evaluation
evaluate.evaluate_clf(l_clf, features, labels)
evaluate.evaluate_clf(k_clf, features, labels)

### Final Machine Algorithm Selection
clf = l_clf

# dump your classifier, dataset and features_list so
# anyone can run/check your results
pickle.dump(clf, open("../data/my_classifier.pkl", "w"))
pickle.dump(my_dataset, open("../data/my_dataset.pkl", "w"))
pickle.dump(my_feature_list, open("../data/my_feature_list.pkl", "w"))