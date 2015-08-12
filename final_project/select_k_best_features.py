# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:28:47 2015

@author: jayantsahewal
"""

import sys
sys.path.append("../tools/")
from feature_format import featureFormat
from feature_format import targetFeatureSplit
from sklearn.feature_selection import SelectKBest

def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    return k_best_features.keys()