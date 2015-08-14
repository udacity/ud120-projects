# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:07:16 2015

@author: jayantsahewal

This module provides methods for machine learning to use in building POI 
prediction model from Enron dataset

This module has following functions:
get_scores: calculates precision and recall for a classifier
get_k_features: selects k best features usking sklearn
get_logReg_optimizer: pipeline and parameters for Logistic Regression
get_kNeighbor_optimizer: pipeline and parameters for K Neighbors Classifier

"""

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def get_scores(clf, dataset, feature_list, folds = 1000):

  """
  calculates precision and recall for a given classifier, data dictionary
  and a list of features
    
  Args:
    clf: classifier for which scores need to be calculated
    dataset: data dictionary for enron
    feature_list: a list of features with first feature as target label
    folds: Number of re-shuffling & splitting iterations

  Returns:
    returns precision and recall
  """
  
  data = featureFormat(dataset, feature_list, sort_keys = True)
  labels, features = targetFeatureSplit(data)
  cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
  true_negatives = 0
  false_negatives = 0
  true_positives = 0
  false_positives = 0
  for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
      features_train.append( features[ii] )
      labels_train.append( labels[ii] )
    for jj in test_idx:
      features_test.append( features[jj] )
      labels_test.append( labels[jj] )
    
    ### fit the classifier using training set, and test on test set
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
      if prediction == 0 and truth == 0:
        true_negatives += 1
      elif prediction == 0 and truth == 1:
        false_negatives += 1
      elif prediction == 1 and truth == 0:
        false_positives += 1
      elif prediction == 1 and truth == 1:
        true_positives += 1
      else:
        print "Warning: Found a predicted label not == 0 or 1."
        print "All predictions should take value 0 or 1."
        print "Evaluating performance for processed predictions:"
        break
  try:
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    return precision, recall
  except:
    print "Got a divide by zero when trying out:", clf
      

def get_k_best_features(data_dict, features_list, k):

  """
  runs scikit-learn's SelectKBest feature selection to get k best features
    
  Args:
    data_dict: data dictionary for enron
    feature_list: a list of features with first feature as target label
    k: Number of best features which need to be selected

  Returns:
    returns a list of k best features and list of lists where inner list's 
    first element is feature and the second element is feature score
  """

  data = featureFormat(data_dict, features_list)
  labels, features = targetFeatureSplit(data)

  k_best = SelectKBest(k=k)
  k_best.fit(features, labels)
  scores = k_best.scores_
  unsorted_pairs = zip(features_list[1:], scores)
  sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
  k_best_features = dict(sorted_pairs[:k])
  return k_best_features.keys(), map(list, sorted_pairs)

def get_logReg_optimizer(reoptimize = False):
  """
  Make a pipeline and parameters dictionary for cross-validated grid search 
  for the Logistic Regression classifier. 
  
  The pipeline has following steps:
  1. Scales the features between 0-1 using MinMaxScaler()
  2. Selects K Best features using Anova F-value scoring for classification.
  3. Uses KBest features to reduce dimensionality using PCA
  4. Uses the resulting PCA components in Logistic Regression Classifier.
  
  Parameters dictionary include:
  SelectKBest:        
    1. k: Number of K Best features to select.
  PCA:
    1. n: Number of PCA components to retain.
    2. whiten: Boolean value whether to whiten the features during PCA.
  LogisticRegression:
    1. C: Value of the regularization constraint.
    2. class_weight: Over-/undersamples the samples of each class.
    3. tol: Tolerance for stopping criteria
  
  Args:
    reoptimize: Boolean value whether to return a range of parameters for
    reoptimization or return the already optimized parameters dictionary.
    REOPTIMIZATION MAY TAKE A LONG TIME.

  Returns:
    A dictionary of parameters to pass into an sklearn grid search pipeline. 
    Default parameters include only the optimized parameters found through 
    multiple runs.
  """
  
  pipe = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                         ('select', SelectKBest(score_func=f_classif)), 
                          ('pca', PCA()), 
                          ('classifier', LogisticRegression())
                          ])
  
  best_parameters = {'select__k': ['all'],
                     'pca__n_components': [3], 
                     'pca__whiten': [False],
                     'classifier__class_weight': ['auto'],
                     'classifier__tol': [1e-2],
                     'classifier__C': [1e-1]
                     }

  search_parameters = {'select__k': [5, 9, 13, 17, 'all'],
                'pca__n_components': [0.5, 1, 2, 3],
                'pca__whiten': [True, False],                
                'classifier__C': [1e-1, 1, 1e2, 1e4, 1e8, 1e16],
                'classifier__class_weight': ['auto'],
                'classifier__tol': [1e-1, 1e-2, 1e-3, 1e-4]
                }
  
  if reoptimize:
    return pipe, search_parameters
  else:
    return pipe, best_parameters

def get_kNeighbor_optimizer(reoptimize = False):
  """
  Make a pipeline and parameters dictionary for cross-validated grid search 
  for K Neighbor classifier. 
  
  The pipeline has following steps:
  1. Scales the features between 0-1 using MinMaxScaler()
  2. Selects K Best features using Anova F-value scoring for classification.
  3. Uses KBest features to reduce dimensionality using PCA
  4. Uses the resulting PCA components in K Neighbor Classifier.
  
  Parameters dictionary include:
  SelectKBest:        
    1. k: Number of K Best features to select.
  PCA:
    1. n: Number of PCA components to retain.
    2. whiten: Boolean value whether to whiten the features during PCA.
  KNeighborsClassifier:
    1. leaf_size: Leaf size passed to BallTree or KDTree.
    2. n_neighbors: Number of neighbors to use by default for k_neighbors queries.
    3. p: Power parameter for the Minkowski metric.
    4. weights: weight function used in prediction.
    5. algorithm: Algorithm used to compute the nearest neighbors.
  
  Args:
    reoptimize: Boolean value whether to return a range of parameters for
    reoptimization or return the already optimized parameters dictionary.
    REOPTIMIZATION MAY TAKE A LONG TIME.

  Returns:
    A dictionary of parameters to pass into an sklearn grid search pipeline. 
    Default parameters include only the optimized parameters found through 
    multiple runs.
  """
  
  pipe = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                         ('select', SelectKBest(score_func=f_classif)), 
                          ('pca', PCA()), 
                          ('classifier', KNeighborsClassifier())
                          ])
  
  best_parameters = {'select__k': [4],
                     'pca__n_components': [0.5], 
                     'pca__whiten': [False],
                     'classifier__leaf_size': [64],
                     'classifier__n_neighbors': [1], 
                     'classifier__p': [3], 
                     'classifier__weights': ['uniform'], 
                     'classifier__algorithm' : ['auto']
                     }

  search_parameters = {'select__k': [4, 5, 10, 11, 17, 'all'],
                'pca__n_components': [0.5, 1, 2, 3],
                'pca__whiten': [True, False],                
                'classifier__leaf_size': [1, 2, 4, 8, 16, 32, 64],
                'classifier__n_neighbors': [1, 2, 3, 4, 5],
                'classifier__p': [1, 2, 3, 4, 5],                  
                'classifier__weights': ['distance', 'uniform'],
                'classifier__algorithm' : ['auto', 'ball_tree', 
                                           'kd_tree', 'brute']
                }
  
  if reoptimize:
    return pipe, search_parameters
  else:
    return pipe, best_parameters