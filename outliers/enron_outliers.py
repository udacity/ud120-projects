#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

data_dict.pop('TOTAL',0)

#print data_dict.keys()
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

pos = 0
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    #print pos, salary, bonus, bonus/salary
    matplotlib.pyplot.scatter( salary, bonus )

keys = data_dict.keys()

for key in keys:
    print key, data_dict[key]['salary'], data_dict[key]['bonus']




