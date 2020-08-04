#!/usr/bin/python3

import pickle
import sys
import matplotlib.pyplot

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
file_path = "../final_project/final_project_dataset.pkl"
data_dict = pickle.load(open(file_path, "rb"), fix_imports=True)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
