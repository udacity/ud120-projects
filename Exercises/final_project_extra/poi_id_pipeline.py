#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

import pandas as pd

from copy import copy
from evaluate import get_scores

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

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

### Classifier
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition

df = pd.DataFrame.from_dict(data_dict, orient='index')
del df['email_address']

df = df.replace('NaN', 0)
df = df.apply(lambda x: x.fillna(0), axis=0)

features = df.drop('poi', axis=1).astype(float)
labels = df['poi']
    
labels = labels[features.abs().sum(axis=1) != 0]
features = features[features.abs().sum(axis=1) != 0]

sk_fold = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1)

pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                             ('selection', SelectKBest(score_func=f_classif)),
                             ('reducer', PCA()),
                             ('classifier', LogisticRegression())
                             ])
                             
params = {'reducer__n_components': [.5], 
            'reducer__whiten': [False],
            'classifier__class_weight': ['auto'],
            'classifier__tol': [1e-64], 
            'classifier__C': [1e-3],
            'selection__k': [17]
            }

scoring_metric = 'recall'

grid_searcher = GridSearchCV(pipeline, param_grid=params, cv=sk_fold, 
                             n_jobs=-1, scoring=scoring_metric, verbose=0)
                             
grid_searcher.fit(features, y=labels)

mask = grid_searcher.best_estimator_.named_steps['selection'].get_support()
top_features = [x for (x, boolean) in zip(features, mask) if boolean]
n_pca_components = grid_searcher.best_estimator_.named_steps['reducer'].n_components_

print "Cross-validated {0} score: {1}".format(scoring_metric, grid_searcher.best_score_)
print "{0} features selected".format(len(top_features))
print "Reduced to {0} PCA components".format(n_pca_components)
###################
# Print the parameters used in the model selected from grid search
print "Params: ", grid_searcher.best_params_ 
###################

clf = grid_searcher.best_estimator_

features.insert(0, 'poi', labels)
jay_data_dict = features.T.to_dict()

test_classifier(clf, jay_data_dict, total_features_list)




#cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
#
#true_negatives = 0
#false_negatives = 0
#true_positives = 0
#false_positives = 0
#for train_idx, test_idx in cv: 
#  features_train = []
#  features_test  = []
#  labels_train   = []
#  labels_test    = []
#  for ii in train_idx:
#    features_train.append( features[ii] )
#    labels_train.append( labels[ii] )
#    for jj in test_idx:
#      features_test.append( features[jj] )
#      labels_test.append( labels[jj] )
#
#estimator.fit(features_train, labels_train)
#
#predictions = estimator.predict(features_test)
#
#for prediction, truth in zip(predictions, labels_test):
#  if prediction == 0 and truth == 0:
#    true_negatives += 1
#  elif prediction == 0 and truth == 1:
#    false_negatives += 1
#  elif prediction == 1 and truth == 0:
#    false_positives += 1
#  elif prediction == 1 and truth == 1:
#    true_positives += 1
#  else:
#    print "Warning: Found a predicted label not == 0 or 1."
#    print "All predictions should take value 0 or 1."
#    print "Evaluating performance for processed predictions:"
#    break
#
#try:
#  total_predictions = true_negatives + false_negatives + false_positives + true_positives
#  accuracy = 1.0*(true_positives + true_negatives)/total_predictions
#  precision = 1.0*true_positives/(true_positives+false_positives)
#  recall = 1.0*true_positives/(true_positives+false_negatives)
#  f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
#  f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
#  print estimator
#  print accuracy
#  print precision
#  print ""
#except:
#  print "Got a divide by zero when trying out:", clf





#sk_fold = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1)
#
#pipeline = get_LogReg_pipeline()
#params = get_LogReg_params(full_search_params=False)
#
#scoring_metric = 'recall'
#grid_searcher = GridSearchCV(pipeline, param_grid=params, cv=sk_fold, 
#                             n_jobs=-1, scoring=scoring_metric, verbose=0)
#
#grid_searcher.fit(features, y=labels)
#
#mask = grid_searcher.best_estimator_.named_steps['selection'].get_support()
##top_features = [x for (x, boolean) in zip(X_features, mask) if boolean]
#n_pca_components = grid_searcher.best_estimator_.named_steps['reducer'].n_components_
#
#print "Cross-validated {0} score: {1}".format(scoring_metric, grid_searcher.best_score_)
##print "{0} features selected".format(len(top_features))
#print "Reduced to {0} PCA components".format(n_pca_components)
####################
## Print the parameters used in the model selected from grid search
#print "Params: ", grid_searcher.best_params_
#
#clf = grid_searcher.best_estimator_
#    
#test_classifier(clf, final_data, total_features_list)


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