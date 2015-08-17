# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:07:16 2015

@author: jayantsahewal

This module provides methods for data manipulation to use in building POI 
prediction model from Enron data

This module has following functions:
get_features: extracts features from data
add_features_totals: creates totals features for compensation and poi interaction
count_values: counts total and valid values
remove_outlier: removes outliers from data
targetFeatureSplitPandas: Splits data dictionary into features dataframe and labels dataframe
"""

import numpy as np
import pandas as pd


def get_features(data_dict, target_label):
  """
  Extracts features from Enron dataset with first features as target label.
    
  Args:
    data_dict: Data dictionary for the enron dataset
    target_label: target label i.e. poi identifier for enron dataset

  Returns:
    list with features from Enron dataset with first features as target label
  """

  ### List of persons
  persons_list = data_dict.keys()
    
  ### total list of features extracted from project dataset
  total_features = {}
  for person in persons_list:
    for key in data_dict[person].keys():
      if data_dict[person][key] == "NaN":
        value = "NaN"
      else:
        value = type(data_dict[person][key])
      if key not in total_features.keys():
        total_features[key] = []    
      if value not in total_features[key]:
        total_features[key].append(value)
    
  ### find out feature which is of string type to exclude it
  exclude_features = []          
  for feature in total_features:
    if type('string') in total_features[feature]:
      if feature not in exclude_features:
        exclude_features.append(feature)
    
  ### Now create a comprehensive list of features excluding string features
  ### and first feature as poi which is the label
  my_features_list = [] + [target_label]
  for feature in total_features:
    if feature not in my_features_list and feature not in exclude_features:
      my_features_list.append(feature)
    
  return my_features_list


def add_features_totals(data_dict, features_list):
  """
  creates totals features for compensation and poi interaction
    
  Args:
    data_dict: Data dictionary for the enron dataset
    features_list: List of features

  Returns:
    data_dict: dictionary with added features
    features_list: list of features with added features
  """
  
  data_dict = data_dict
  features_list = features_list
  
  financial_fields = ['total_stock_value', 'total_payments']
  for record in data_dict:
    person = data_dict[record]
    is_valid = True
    for field in financial_fields:
      if person[field] == 'NaN':
        is_valid = False
    if is_valid:
      person['total_compensation'] = sum([person[field] for field in financial_fields])
    else:
      person['total_compensation'] = 'NaN'
  
  features_list += ['total_compensation']

  """ add total poi interaction """
  
  email_fields = ['shared_receipt_with_poi', 'from_this_person_to_poi', 'from_poi_to_this_person']
  for record in data_dict:
    person = data_dict[record]
    is_valid = True
    for field in email_fields:
      if person[field] == 'NaN':
        is_valid = False
    if is_valid:
      person['total_poi_interaction'] = sum([person[field] for field in email_fields])
    else:
      person['total_poi_interaction'] = 'NaN'
  
  features_list += ['total_poi_interaction']
  
  return data_dict, features_list

    
def count_values(data_dict, features):
  """
  Counts valid (non-NaN) and total values for each feature
    
  Args:
    data_dict: Data dictionary for the enron dataset
    features: List of features

  Returns:
    list with features, valid counts and total counts
  """

  feature_counts = []
  for feature in features:
    valid_count = 0
    total_count = 0
    for record in data_dict:
      person = data_dict[record]
      total_count += 1
      if person[feature] != 'NaN':
        valid_count += 1
    feature_counts.append([feature, valid_count, total_count])
        
  return feature_counts


def remove_outlier(data_dict, keys):
  """
  removes a list of keys from a dictionary
    
  Args:
    data_dict: Data dictionary for the enron dataset
    keys: List of names which need to be removed

  Returns:
    Dictionary after removing the outliers
  """

  for key in keys:
    data_dict.pop(key, 0)
  
  return data_dict

def targetFeatureSplitPandas(data_dict):
  """
  Splits data dictionary into features dataframe and labels dataframe.
  It also removes email address which is a string feature and is of no value
  in the further analysis
    
  Args:
    data_dict: data dictionary with Enron features and target labels.

  Returns:
    One pandas dataframe of all features columns.
    One pandas dataframe of the target 'poi' labels.  
  """
  
  # Convert to pandas dataframe for the data shaping phase.
  df = pd.DataFrame.from_dict(data_dict, orient='index')
  
  # Remove email_address strings since they won't be used at all.
  del df['email_address']
  
  # replace Pandas dataframe with 0's in the place of NaN and/or Inf/-Inf.
  df = df.replace('NaN', 0)
  df = df.replace([np.inf, -np.inf], 0)
  df = df.apply(lambda x: x.fillna(0), axis=0)
  
  # extract features and labels from the Pandas dataframe
  features = df.drop('poi', axis=1).astype(float)
  labels = df['poi']
  
  # Remove rows which absolute values sum up to zero from labels and features.
  labels = labels[features.abs().sum(axis=1) != 0]
  features = features[features.abs().sum(axis=1) != 0]
  
  return labels, features
  
def combineLabelsFeatures(labels, features):
  """
  combines features and labels pandas dataframes to data_dict format
  
  Args:
    features: Pandas dataframe containing features
    labels: Pandas dataframe containing labels
    
  Returns:
    data_dict: a data dictionary
  """  
  features.insert(0, 'poi', labels)
    
  data_dict = features.T.to_dict()

  return data_dict
  
  
  