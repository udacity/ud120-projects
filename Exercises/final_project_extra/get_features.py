# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:56:00 2015

@author: jayantsahewal
"""

def features(data_dict, target_label):
    
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