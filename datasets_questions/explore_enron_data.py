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

# ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus',
# 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses',
# 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'poi', 'director_fees', 'deferred_income',
# 'long_term_incentive', 'email_address', 'from_poi_to_this_person']


print len(enron_data)

poi = 0
total_payments = 0
for index in range(0, len(enron_data)):
    person_name = enron_data.keys()[index]
    if enron_data[person_name]['poi']:
        poi = poi + 1
        if enron_data[person_name]['total_payments'] == 'NaN':
            total_payments = total_payments + 1

print poi
print total_payments

# Find total stock value for James Prentice
# print enron_data["PRENTICE JAMES"]['total_stock_value']

# How many email messages do we have from Wesley Colwell to persons of interest?
# print enron_data["COLWELL WESLEY"]['from_this_person_to_poi']

# print enron_data["SKILLING JEFFREY K"]['total_payments']
# print enron_data["LAY KENNETH L"]['total_payments']
# print enron_data["FASTOW ANDREW S"]['total_payments']

# salary = 0
# email = 0
# for index in range(0, len(enron_data)):
#     person_name = enron_data.keys()[index]
#     if enron_data[person_name]['salary'] != 'NaN':
#         salary = salary + 1
#     if enron_data[person_name]['email_address'] != 'NaN':
#         email = email + 1
#
# print salary
# print email

no_payments = 0
for index in range(0, len(enron_data)):
    person_name = enron_data.keys()[index]
    if enron_data[person_name]['total_payments'] == 'NaN':
        no_payments = no_payments + 1

print no_payments
print float(no_payments*100)/float(len(enron_data))