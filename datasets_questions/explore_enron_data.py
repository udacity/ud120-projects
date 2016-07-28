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

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

def first_part():
    print (len(enron_data.keys()))
    print (len(enron_data["PIPER GREGORY F"]))
    
    poi_count=0
    for p in enron_data.keys():
        if enron_data[p]["poi"]==1:
            poi_count+=1
            
    print (poi_count)
    
    print (enron_data["Prentice James".upper()])
    print (enron_data["Colwell Wesley".upper()]["from_this_person_to_poi"])
    print ("----")
    print (enron_data["SKILLING JEFFREY K".upper()]["total_payments"])
    print (enron_data["FASTOW ANDREW S".upper()]["total_payments"])
    print (enron_data["LAY KENNETH L".upper()]["total_payments"])
    print ("---")
    salary = 0
    email_address = 0
    for p in enron_data.keys():
        if enron_data[p]["salary"]!='NaN':
            salary+=1
        if enron_data[p]["email_address"]!="NaN":
            email_address+=1
    
    print (salary)
    print (email_address)
    
def missing_pois():
    missing_poi = [poi for poi in enron_data.keys() if enron_data[poi]["total_payments"]=="NaN"]
    print ("missing payment %d" % len(missing_poi))
    percent = (len(missing_poi)/len(enron_data))*100 
    print ("percent of missing %.03f" % percent)
    
    missing_poi = [poi for poi in enron_data.keys() if (enron_data[poi]["total_payments"]=="NaN" and enron_data[poi]["poi"]==True)]
    print ("missing payment poi %d" % len(missing_poi))
    percent = (len(missing_poi)/len(enron_data))*100
    print ("percent of missing poi payment %.03f" % percent)
    
    print ("number of pois in dataset %d" % len([poi for poi in enron_data.keys() if enron_data[poi]["poi"]==True]))
    
if __name__ == '__main__':
    first_part()
    missing_pois()


