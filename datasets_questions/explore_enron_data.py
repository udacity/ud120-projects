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

# {'salary': 243293, 'to_messages': 1045, 'deferral_payments': 'NaN', 'total_payments': 288682, 'exercised_stock_options': 5538001, 'bonus': 1500000, 'restricted_stock': 853064, 'shared_receipt_with_poi': 1035, 'restricted_stock_deferred': 'NaN', 'total_stock_value': 6391065, 'expenses': 34039, 'loan_advances': 'NaN', 'from_messages': 32, 'other': 11350, 'from_this_person_to_poi': 21, 'poi': True, 'director_fees': 'NaN', 'deferred_income': -3117011, 'long_term_incentive': 1617011, 'email_address': 'kevin.hannon@enron.com', 'from_poi_to_this_person': 32}

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


values = enron_data.values()
pois = [elem for elem in values if elem['poi'] == True]

def filterByFeature(feature):
    return [item[feature] for item in values]

sortedTotalPayments = sorted(enron_data, key = lambda name: enron_data[name]["total_payments"])

# for key in sortedTotalPayments:
#     print key, enron_data[key]["total_payments"]

nan_payment = len([elem for elem in values if (elem["total_payments"] == 'NaN' and elem["poi"] == True)])
total = len([elem for elem in filterByFeature("poi") if elem == True])
print "total_payments is NaN", nan_payment
print "total set ", total
print "%", (float(nan_payment) / total) * 100
