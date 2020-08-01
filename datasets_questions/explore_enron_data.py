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
import numpy as np

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print(len(enron_data))
print(len(list(enron_data.values())[0]))

count = 0
for person_name in enron_data.keys():
	if(enron_data[person_name]["poi"]==1):
		count = count+1
print(count)

total_poi = 0
with open('../final_project/poi_names.txt', 'r') as file:
	for line in file:
		if('\(y\)' or '\(n\)' in line):
			total_poi= total_poi+1
print(total_poi)
file.close()
print("Net Stock value of James Prentice: ", enron_data['PRENTICE JAMES']['total_stock_value'])
print("Wesley Colwell to POI emails: ", enron_data['COLWELL WESLEY']['from_this_person_to_poi'])
print("Stock options of Jeffrey Skilling: ", enron_data['SKILLING JEFFREY K']['exercised_stock_options'])

most_value_taken = max([(enron_data[person_name]['total_payments']) for person_name in ("LAY KENNETH L", "SKILLING JEFFREY K", "FASTOW ANDREW S")])
print(most_value_taken)

salaries_not_nan = 0
known_emails = 0
total_payments_not_nan = 0
total_payments_not_nan_poi = 0
for person_name in enron_data:
	if not np.isnan(float(enron_data[person_name]['salary'])):
		salaries_not_nan += 1
	if(enron_data[person_name]['email_address'] != 'NaN'):
		known_emails+=1
	if np.isnan(float(enron_data[person_name]['total_payments'])):
		total_payments_not_nan +=1
		if np.isnan(enron_data[person_name]["poi"]==1 ):
			total_payments_not_nan_poi += 1

print('Salaries available:: ', salaries_not_nan)
print('Available emails: ', known_emails)
print('Number Percentage people NaN -> their total payments: ',total_payments_not_nan, total_payments_not_nan*100/len(enron_data))
print('Number and Percentage Pois NaN ->  their total payments: ',total_payments_not_nan_poi, total_payments_not_nan_poi*100/count)
