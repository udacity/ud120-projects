#!/usr/bin/python
# coding=utf-8

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
import  numpy as np

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print enron_data

print '# of people in E+F dataset: ', len(enron_data)

features = {}
for i, (name, personal_data) in enumerate(enron_data.items()):
    for i, (feature, value) in enumerate(personal_data.items()):
        if not feature in features:
            features[feature] = 1
        else:
            features[feature] += 1
print '# of features in E+F dataset: ', len(features)


n_poi = 0
for i, (name, personal_data) in enumerate(enron_data.items()):
    if enron_data[name]["poi"] == 1:
        n_poi += 1
print '# of poi in E+F dataset: ', n_poi


with open('../final_project/poi_names.txt', 'r') as f:
    poi_names = f.read().splitlines()[2:]
    print '# of POI: ', len(poi_names)

tsv = 0
for i, (name, personal_data) in enumerate(enron_data.items()):
    if name == 'PRENTICE JAMES':
        tsv += enron_data[name]["total_stock_value"]
print 'TSV of James Prentice: ', tsv

n_to_poi = 0
for i, (name, personal_data) in enumerate(enron_data.items()):
    if name == 'COLWELL WESLEY':
        n_to_poi += enron_data[name]["from_this_person_to_poi"]
print 'Wesley Colwell to POI: ', n_to_poi

ex_stock = 0
for i, (name, personal_data) in enumerate(enron_data.items()):
    if name == 'SKILLING JEFFREY K':
        ex_stock += enron_data[name]["exercised_stock_options"]
print 'Jeffrey K Skilling exercised stock options: ', ex_stock

payments = {}
for i, (name, personal_data) in enumerate(enron_data.items()):
    if name in ['SKILLING JEFFREY K', 'FASTOW ANDREW S', 'LAY KENNETH L']:
        payments[name] = enron_data[name]["total_payments"]
print payments

n_has_salary = 0
for i, (name, personal_data) in enumerate(enron_data.items()):
    if enron_data[name]["salary"] != 'NaN':
        n_has_salary += 1
print '# of people has salary: ', n_has_salary

n_has_email = 0
for i, (name, personal_data) in enumerate(enron_data.items()):
    if enron_data[name]["email_address"] != 'NaN':
        n_has_email += 1
print '# of people has email: ', n_has_email

n_missing_payments = 0
for i, (name, personal_data) in enumerate(enron_data.items()):
    if enron_data[name]["total_payments"] == 'NaN':
        n_missing_payments += 1
print '# of people do not have total payments: ', n_missing_payments
print '% of people do not have total payments: ', (n_missing_payments * 1.0) / len(enron_data) * 100

n_missing_payments = 0
for i, (name, personal_data) in enumerate(enron_data.items()):
    if enron_data[name]["total_payments"] == 'NaN' and enron_data[name]["poi"] == 1:
        n_missing_payments += 1
print '# of poi do not have total payments: ', n_missing_payments
print '% of poi do not have total payments: ', (n_missing_payments * 1.0) / (n_poi*1.0) * 100


# adding 10 poi without total payments => ptg of people missing: 20%; ptg of POI missing: 36%.
# Missing total payments could be used as a clue for supervised classification algorithm.
# However, it would introduce biases. suppose you use your POI detector
# to decide whether a new, unseen person is a POI, and that person isn’t on the spreadsheet.
# Then all their financial data would contain “NaN” but the person is very likely not a POI
# (there are many more non-POIs than POIs in the world, and even at Enron)
# --you’d be likely to accidentally identify them as a POI, though!

# if your data are coming from different sources for different classes,
# It can easily lead to the type of bias or mistake that we showed here.
# Solution 1: used only email data
# --in that case, discrepancies in the financial data wouldn’t matter because financial features aren’t being used.
# Solution 2: estimating how much of an effect these biases can have on your final answer.



#for i, (name, personal_data) in enumerate(enron_data.items()):
#    print name