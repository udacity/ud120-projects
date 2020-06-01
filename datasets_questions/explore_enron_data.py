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

count = 0
for person in enron_data:
    if enron_data[person]['salary'] != "NaN":
        count += 1
print(count)

count = 0
for person in enron_data:
    if enron_data[person]['email_address'] != "NaN":
        count += 1

count = 0
num = 0
without = 0
for person in enron_data:
    if enron_data[person]['poi'] == True:
        count +=1
        if enron_data[person]['total_payments'] != "NaN":
            num += 1
        elif enron_data[person]['total_payments'] == "NaN":
            without += 1
print('Total Number of People: %s' %count)
print('Number WITH Payments : %s' % num)
print('Total number of people without data: %s' %(count-num))
print('Percentage: %s' %(1.0*without/count*100))