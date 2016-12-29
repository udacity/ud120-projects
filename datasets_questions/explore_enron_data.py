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


#data = enron_data['PRENTICE JAMES']
#data = enron_data['COLWELL WESLEY'] enron_data['LAY KENNETH L']
#data = enron_data['FASTOW ANDREW S'] enron_data['SKILLING JEFFREY K']['exercised_stock_options']


##cnt = 0
##
##for v in enron_data:
##    if enron_data[v]['poi'] == True:
##        cnt+=1
##
##print cnt

##count_salary = 0
##count_email = 0
##for key in enron_data.keys():
##    if enron_data[key]['salary'] != 'NaN':
##        count_salary+=1
##    if enron_data[key]['email_address'] != 'NaN':
##        count_email+=1
##print count_salary
##print count_email

count_NaN_tp = 0
for key in enron_data.keys():
    if enron_data[key]['total_payments'] == 'NaN'and enron_data[key]['poi'] == True:
        count_NaN_tp+=1
print count_NaN_tp
print float(count_NaN_tp)/len(enron_data.keys())
