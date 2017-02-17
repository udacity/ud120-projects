#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL",0)
data = featureFormat(data_dict, features)


#==============================================================================
# max_bonus=0
# for l in data:
#     if l[1]>max_bonus:
#         max_bonus=l[1]
#         
# for d in data_dict.keys():
#     if data_dict[d]["bonus"]== max_bonus and d!="TOTAL":
#         print (data_dict[d])
#==============================================================================

for d in data_dict.keys():
    if float(data_dict[d]["bonus"])>5*10**6 and d!="TOTAL" and float(data_dict[d]["salary"])>10**6:
        print (d)

### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

