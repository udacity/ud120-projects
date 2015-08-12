# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 03:10:23 2015

@author: jayantsahewal
"""

def count_valid_values(data_dict, features):
  """ counts the number of non-NaN (valid) values for each feature,
      total values for each feature and appends these values to each
      feature. so, the returned list has features, features scores,
      valid counts and total counts
  """
  for feature in features:
    feature_name = feature[0]
    feature.append(0)
    feature.append(0)
    for record in data_dict:
      person = data_dict[record]
      feature[3] += 1
      if person[feature_name] != 'NaN':
        feature[2] += 1
        
  return features