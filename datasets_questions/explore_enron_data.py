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

# load dataset into dataframe
import pandas as pd

df = pd.DataFrame(enron_data)
print(df.shape)

d = df.transpose()
print(d.head())
#df['poi']

#poi_count =
#print(df.loc['poi'] == True)
#print("%d people are persons of interest" % poi_count)

with open('../final_project/poi_names.txt', 'r') as f:
    count = 0
    for line in f.readlines():
        if "(y)" in line or "(n)" in line:
            count += 1
    print("there were in total %d persons of interest" % count)




