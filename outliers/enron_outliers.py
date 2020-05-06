#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
# remove Total from dictionary
print len(data_dict)
data_dict.pop( 'TOTAL', 0 )
print len(data_dict)
# format as numpy
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


### find outliers
for (key, value) in data_dict.items():
   # Check if key is even then add pair to new dictionary
   if value['salary'] > 1000000 and value['bonus'] > 5000000 and value['bonus'] > value['salary']:
       print key, value['salary'], value['bonus']