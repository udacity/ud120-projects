#!/usr/bin/python

"""
    starter code for exploring the Enron dataset (emails + finances)
    loads up the dataset (pickled dict of dicts)
    
    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }
    
    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:
    
    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
    """

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

count_poi = 0
count_salary = 0
count_email = 0
count_payments = 0
count_poi_payments = 0

for key in enron_data.keys():
    if enron_data[key]["poi"] == True:
        count_poi += 1




for key, value in enron_data.iteritems():
    if key == "SKILLING JEFFREY K":
        print "Skilling"
        print value['total_payments']
    if key == "FASTOW ANDREW S":
        print "Fastow"
        print value['total_payments']
    if key == "LAY KENNETH L":
        print "Lay"
        print value['total_payments']

for key, value in enron_data.iteritems():
    for k, v in value.iteritems():
        if k == 'salary':
            if v != 'NaN':
                count_salary += 1
        if k == 'email_address':
            if v != 'NaN':
                count_email +=1
        if k == 'total_payments':
            if v != 'NaN':
                count_payments +=1

for key in enron_data.keys():
    if enron_data[key]["poi"] == True:
        if enron_data[key]["total_payments"] != "NaN":
            count_poi_payments += 1

print "printing counts...."
print count_poi
print count_salary
print count_email
print count_payments
print count_poi_payments
print len(enron_data)