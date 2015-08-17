# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:07:16 2015

@author: jayantsahewal

This module provides methods for machine learning to use in building POI 
prediction model from Enron dataset

One Global variable scoring_metric has been defined which will decide which
score needs to be the best while optimizing the parameters.

This module has following functions:
get_scores: calculates precision and recall for a classifier
get_k_features: selects k best features usking sklearn
get_logReg_optimizer: pipeline and parameters for Logistic Regression
get_kNeighbor_optimizer: pipeline and parameters for K Neighbors Classifier
get_svm_optimizer: pipeline and parameters for Support Vector Machine
get_rForest_optimizer: pipeline and paramters for Random Forest


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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

### Since we need to optimize our parameters for both recall and precision
### better than 0.3, so I have chosen F1 as my scoring metric for GridSearchCV
### which weighs recall and precision equally.
scoring_metric = 'f1'


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
                          ('clf', LogisticRegression())
                          ])
  
  best_parameters = {'select__k': ['all'],
                     'pca__n_components': [3], 
                     'pca__whiten': [False],
                     'clf__class_weight': ['auto'],
                     'clf__tol': [1e-2],
                     'clf__C': [1e-1]
                     }

  search_parameters = {'select__k': [5, 9, 13, 17, 'all'],
                       'pca__n_components': [0.5, 1, 2, 3],
                       'pca__whiten': [True, False],                                
                       'clf__C': [1e-1, 1, 1e2, 1e4, 1e8, 1e16],                
                       'clf__class_weight': ['auto'],                
                       'clf__tol': [1e-1, 1e-2, 1e-3, 1e-4]
                       }
  
  if reoptimize:
    return pipe, search_parameters, scoring_metric
  else:
    return pipe, best_parameters, scoring_metric

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
                          ('clf', KNeighborsClassifier())
                          ])
  
  best_parameters = {'select__k': [4],
                     'pca__n_components': [0.5], 
                     'pca__whiten': [False],
                     'clf__leaf_size': [64],
                     'clf__n_neighbors': [1], 
                     'clf__p': [3], 
                     'clf__weights': ['uniform'], 
                     'clf__algorithm' : ['auto']
                     }

  search_parameters = {'select__k': [4, 5, 10, 11, 17, 'all'],
                'pca__n_components': [0.5, 1, 2, 3],
                'pca__whiten': [True, False],                
                'clf__leaf_size': [1, 2, 4, 8, 16, 32, 64],
                'clf__n_neighbors': [1, 2, 3, 4, 5],
                'clf__p': [1, 2, 3, 4, 5],                  
                'clf__weights': ['distance', 'uniform'],
                'clf__algorithm' : ['auto', 'ball_tree', 
                                           'kd_tree', 'brute']
                }
  
  if reoptimize:
    return pipe, search_parameters, scoring_metric
  else:
    return pipe, best_parameters, scoring_metric


def get_svm_optimizer(reoptimize = False):
  """
  Make a pipeline and parameters dictionary for cross-validated grid search 
  for Support Vector Machines. 
  
  The pipeline has following steps:
  1. Scales the features between 0-1 using MinMaxScaler()
  2. Selects K Best features using Anova F-value scoring for classification.
  3. Uses KBest features to reduce dimensionality using PCA
  4. Uses the resulting PCA components in Support Vector Machines classifier.
  
  Parameters dictionary include:
  SelectKBest:        
    1. k: Number of K Best features to select.
  PCA:
    1. n: Number of PCA components to retain.
    2. whiten: Boolean value whether to whiten the features during PCA.
  SVC:
    1. C: Regularization constraint.
    2. class_weight: Over-/undersamples the samples of each class
    3. tol: Tolerance for stopping criteria
    4. gamma: Kernel coefficient for 'rbf' kernel
    5: kernel: Specifies the kernel type to be used in the algorithm
  
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
                          ('clf', SVC())
                          ])
  
  best_parameters = {'select__k': [17],
                     'pca__n_components': [2], 
                     'pca__whiten': [True],
                     'clf__C': [1e-2],
                     'clf__gamma': [0.0],
                     'clf__kernel': ['rbf'],
                     'clf__tol': [1e-3],
                     'clf__class_weight': ['auto']
                     }

  search_parameters = {'select__k': [1, 2, 4, 8, 12, 13, 16, 17, 'all'],
                       'pca__n_components': [1, 2, 3, 4, 5, .25, .5, .75, 'mle'],
                       'pca__whiten': [False, True],
                       'clf__C': [1e-5, 1e-2, 1e-1, 1, 10, 1e2, 1e5],
                       'clf__gamma': [0.0],
                       'clf__kernel': ['linear', 'poly', 'rbf'],
                       'clf__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  
                       'clf__class_weight': [{True: 12, False: 1},
                                               {True: 10, False: 1},
                                               {True: 8, False: 1},
                                               {True: 15, False: 1},
                                               {True: 4, False: 1},
                                               'auto', None]
                      }
  
  if reoptimize:
    return pipe, search_parameters, scoring_metric
  else:
    return pipe, best_parameters, scoring_metric


def get_rForest_optimizer(reoptimize = False):
  """
  Make a pipeline and parameters dictionary for cross-validated grid search 
  for Random Forest Classifier. 
  
  The pipeline has following steps:
  1. Scales the features between 0-1 using MinMaxScaler()
  2. Selects K Best features using Anova F-value scoring for classification.
  3. Uses KBest features to reduce dimensionality using PCA
  4. Uses the resulting PCA components in Random Forest classifier.
  
  Parameters dictionary include:
  SelectKBest:        
    1. k: Number of K Best features to select.
  PCA:
    1. n: Number of PCA components to retain.
    2. whiten: Boolean value whether to whiten the features during PCA.
  RandomForestClassifier:
    1. criterion: The function to measure the quality of a split
    2. max_features: The number of features to consider for the best split
    3. max_depth: The maximum depth of the tree
    4. min_samples_split: The minimum samples required to split an internal node
  
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
                          ('clf', RandomForestClassifier())
                          ])
  
  best_parameters = {'select__k': [4],
                     'pca__n_components': [0.25], 
                     'pca__whiten': [True],
                     'clf__criterion': ['gini'],
                     'clf__max_features': ['auto'],
                     'clf__max_depth': [25],
                     'clf__min_samples_split': [2]
                     }

  search_parameters = {'select__k': [3, 4, 7, 8, 11, 12, 15, 16, 17, 'all'],
                       'pca__n_components': [0.25, 0.5, 0.75, 1, 2, 'mle'], 
                       'pca__whiten': [True, False],
                       'clf__criterion': ['gini', 'entropy'],
                       'clf__max_features': [0.25, 0.5, 0.75, 'auto'],
                       'clf__max_depth': [5, 10, 15, 20, 25],
                       'clf__min_samples_split': [2, 4, 8, 16, 32],
                       }
  
  if reoptimize:
    return pipe, search_parameters, scoring_metric
  else:
    return pipe, best_parameters, scoring_metric