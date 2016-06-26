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

salarycount = 0
emailcount = 0
totalpaymentscount = 0
poitotalpaymentscount = 0
count = 0
poicount = 0
totalstockcount = 0

for key in enron_data:
    count += 1
    if enron_data[key]["email_address"] != "NaN":
        emailcount += 1
    if enron_data[key]["salary"] != "NaN":
        salarycount += 1
    if enron_data[key]["total_payments"] == "NaN":
        totalpaymentscount += 1
    if enron_data[key]["poi"] == 1:
        if enron_data[key]["total_stock_value"] == "NaN":
            totalstockcount += 1
    if enron_data[key]["poi"] == 1:
        poicount += 1
        if enron_data[key]["total_payments"] == "NaN":
            poitotalpaymentscount += 1

print (count)
print (totalpaymentscount)
print (emailcount)
print (salarycount)
print (poitotalpaymentscount)
print (poicount)
print (totalstockcount)

print(totalpaymentscount/float(poicount + 10))

# print(enron_data["SKILLING JEFFREY K"]["total_payments"])
# print(enron_data["FASTOW ANDREW S"]["total_payments"])
# print(enron_data["LAY KENNETH L"]["total_payments"])


