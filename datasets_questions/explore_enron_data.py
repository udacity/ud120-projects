#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

count = 0

for person in enron_data:
    count = count+1
    
print "count of emails is %d " %count
#print enron_data["SKILLING JEFFREY K"]

count = 0

def somefiltering(filterDict, *criteria):
    return [key for key in filterDict if all(criterion(filterDict[key]) for criterion in criteria)]

print len(somefiltering(enron_data,  lambda d:d['total_payments'] == 'NaN'))



print len(somefiltering(enron_data, lambda d:d['poi'] == True))
'''
poi = 0
count = 0
for k in enron_data:
    if enron_data[k]['poi'] == True:
        count_poi = poi+1
        if enron_data[k]['total_payments'] == "NaN":
            count_nan += 1
        
         
        #  '== True' can be suppressed
    
    
    

# print count 
print float(count) / len(enron_data)'''