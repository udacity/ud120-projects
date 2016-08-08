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
from pprint import pprint

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

people_count = len(enron_data.keys())
print('Number of people', people_count)

feature_count = len(enron_data.values()[0])
print('Number of features per person', feature_count)

# poi is 1 when true and 0 otherwise, so a summation wield yield the count
poi_count = sum([enron_data[name]['poi'] for name in enron_data])
print('Number of persons of interest', poi_count)

# get total POIs
with open('../final_project/poi_names.txt', 'r') as f:
    for i in range(2):
        f.readline() # first 2 lines don't list the names
    poi_total_count = len(f.readlines())
    print('Total number of POIs', poi_total_count)


# query dataset
name = enron_data.keys()[0]
print("Sample features dict")
pprint(enron_data[name])

prentice = 'PRENTICE JAMES'
print("Prentice's total stock value", enron_data[prentice]['total_stock_value'])

colwell = 'COLWELL WESLEY'
print("Wesley's total from messages to poi", enron_data[colwell]['from_this_person_to_poi'])

skilling = 'SKILLING JEFFREY K'
print("Jeffrey's exercised stock options", enron_data[skilling]['exercised_stock_options'])

# Who took the most money
lay = 'LAY KENNETH L'
fastow = 'FASTOW ANDREW S'
print('Skilling total payments', enron_data[skilling]['total_payments'])
print("Lay's total payments", enron_data[lay]['total_payments'])
print("Fastow's total payments", enron_data[fastow]['total_payments'])

# Number of people with quantified salaries
salary_count = sum([1 if not enron_data[name]['salary'] == 'NaN' else 0 for name in enron_data ])
print('Number of valid salaries', salary_count)

# Number of people with valid email addresses
email_count = sum([1 if not enron_data[name]['email_address'] == 'NaN' else 0 for name  in enron_data])
print('Number of valid emails', email_count)

# Percentage of people without total_payments data
payments_invalid_count = sum(1 if enron_data[name]['total_payments']=='NaN' else 0 for name in enron_data)
print('Number of people without total_payments data', payments_invalid_count)
print('Percentage of those people to the total number of people', payments_invalid_count * 100/float(people_count))

# Percentage of POIs wihout total_payments data
payments_poi_invalid_count = sum(1 if enron_data[name]['total_payments']=='NaN' and enron_data[name]['poi'] else 0 for name in enron_data)
print('Number of POIs without total_payments data', payments_poi_invalid_count)
print('Percentage of POIs with missing total_payments', payments_poi_invalid_count*100/poi_count)