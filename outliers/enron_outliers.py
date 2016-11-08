#!/usr/bin/python
import pandas as pd
import pickle
import sys
import matplotlib.pyplot as pt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from math import isnan

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )


top = None

for key, value in data_dict.iteritems():
    inside_dict = value
    if isinstance(inside_dict['salary'], int) and inside_dict['salary'] > top:
        top = inside_dict['salary']
        left_key = key
    
        
    
print left_key


data_dict.pop('TOTAL',0)
#data_dict.pop('LAY KENNETH L',0)

for key, value in data_dict.iteritems():
    data_entry = value
    for k,v in data_entry.iteritems():
        if isinstance(v, int) and top < v:
            top = v
            right_key = key
        
#print right_key


for key, value in data_dict.iteritems():
    data_entry = value
    
    if isinstance(data_entry['bonus'], int) and data_entry['bonus'] > 5000000 and isinstance(data_entry['salary'], int) and data_entry['salary'] > 1000000:
        print key
        
    
    
        
  
#print df.head(5)    
#print data_dict.keys()

#for name in data_dict.keys():
    #print "the name is %s and the salary is %r" %(name,data_dict[name]['salary'])
#sorted_dict = sorted(data_dict.items(),key=lambda x: x[1]['salary'], reverse = True)

#print sorted_dict




#print data_dict['BAXTER JOHN C']
#print data_dict
features = ["salary", "bonus"]


data = featureFormat(data_dict, features)




### your code below


for point in data:
   # print point
    salary = point[0]
    bonus = point[1]
    pt.scatter(salary, bonus)
    #pt.plot(salary, bonus)

pt.xlabel("Salary")
pt.ylabel("bonus")
pt.show()
