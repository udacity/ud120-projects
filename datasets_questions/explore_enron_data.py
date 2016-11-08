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

import pandas as pd

import numpy as np

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#print  enron_data["SKILLING JEFFREY K"]



for person in enron_data:
    count = count+1
    
#print "count of emails is %d " %count
#print enron_data["SKILLING JEFFREY K"]

count = 0

def somefiltering(filterDict, *criteria):
    return [key for key in filterDict if all(criterion(filterDict[key]) for criterion in criteria)]

#print somefiltering(enron_data,  lambda d:d['exercised_stock_options'] == 'NaN')


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

enron_data.pop('TOTAL',0)
feature = 'salary'

f2 = 'exercised_stock_options'

feature = f2
max_stock = 0
for key,value in enron_data.iteritems():
    if value[feature] > max_stock and isinstance(value[feature],int):
        max_stock = value[feature]
        max_key = key

print max_key
print "maximum value of %s is %d"  %(feature,max_stock)

min_stock = max_stock 
for key,value in enron_data.iteritems():
    if value[feature] < min_stock and isinstance(value[feature],int):
        min_stock = value[feature]
        min_key = key

print min_key
print "minimum value of %s is %d"  %(feature,min_stock)


jaal = max_stock- min_stock
feature_value = 1000000

feature_scaled = float(feature_value-min_stock)/float(jaal)

print "the range of %s is %d" %(feature, jaal)
print "the scaled value of %s is %f" %(feature, feature_scaled)


#print len(somefiltering(enron_data, lambda d:d['poi'] == True))
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