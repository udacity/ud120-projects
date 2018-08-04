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

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

features = set()
for name in enron_data.keys():
    for key in enron_data[name].keys():
        features.add(key) 

print len(features)
print features

print len(list(name for name in enron_data.keys() if enron_data[name]["poi"] == 1))

print enron_data["PRENTICE JAMES"]['total_stock_value']
print enron_data["COLWELL WESLEY"]['from_this_person_to_poi']
print enron_data["SKILLING JEFFREY K"]['exercised_stock_options']
print "Total payments:"
print 'Lay: ', enron_data["LAY KENNETH L"]['total_payments']
print 'Skilling: ', enron_data["SKILLING JEFFREY K"]['total_payments']
print 'Fastow: ', enron_data["FASTOW ANDREW S"]['total_payments']

# print enron_data["PRENTICE JAMES"]

print len(list(name for name in enron_data.keys() if not enron_data[name]['salary'] == 'NaN'))
print len(list(name for name in enron_data.keys() if not enron_data[name]['email_address'] == 'NaN'))

n = len(list(name for name in enron_data.keys() if enron_data[name]['total_payments'] == 'NaN' and 
enron_data[name]['poi'] == 1))
n_poi = len(list(name for name in enron_data.keys() if enron_data[name]['poi'] == 1))
print n, n_poi, 100. * n / n_poi, '%'