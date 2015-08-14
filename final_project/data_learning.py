# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:07:16 2015

@author: jayantsahewal

This module provides methods for machine learning to use in building POI 
prediction model from Enron data

This module has following functions:
get_scores: calculates precision and recall for a classifier
get_k_features: selects k best features usking sklearn
"""

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest


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