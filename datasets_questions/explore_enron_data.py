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
import pandas as pd

original = "../final_project/final_project_dataset.pkl"
destination = "word_data_unix.pkl"


def unix_version(path_origin, path_dest):
    content = ''
    outsize = 0
    with open(path_origin, 'rb') as infile:
        content = infile.read()
        with open(path_dest, 'wb') as output:
            for line in content.splitlines():
                outsize += len(line) + 1
                output.write(line + str.encode('\n'))
    return path_dest

pickled_file = unix_version(original, destination)

infile = open(pickled_file, "rb")
enron_data = pickle.load(infile)

df = pd.DataFrame(enron_data).T
print(df)
