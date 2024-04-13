#!/usr/bin/python3

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

from tkinter.font import names
import joblib

enron_data = joblib.load(open("./final_project/final_project_dataset.pkl", "rb"))

# Print the first 5 items in the enron_data dictionary
for i, (key, value) in enumerate(enron_data.items()):
    if i >= 5:
        break
    print(f"{key}: {value}\n")

# Print the number of data points (people) in the dataset
print(f"Number of data points: {len(enron_data)}")
# Print the number of features for each person in the dataset
print(f"Number of features: {len(list(enron_data.values())[0])}")
# Count the number of POIs in the dataset
print(f"Number of POIs: {sum([1 for person in enron_data.values() if person['poi']== 1])}") # == 1 or == True or only if person['poi'] without == 1
# form the list of POIs names in /final_project/poi_names.txt and print the number of POIs
poi_names = [name for name in open("./final_project/poi_names.txt").read().split("\n") if name not in ["", "http://usatoday30.usatoday.com/money/industries/energy/2005-12-28-enron-participants_x.htm"]]
print(f"Number of POIs: {len(poi_names)}")
print(poi_names)

# List of features for each person in the dataset
first_person_features = list(enron_data.values())[0]
# List of names in the dataset
names_enron_data = list(enron_data.keys())


# Print the total value of the stock belonging to James Prentice
stock_features = [feature for feature in first_person_features.keys() if 'stock' in feature.lower() and 'total' in feature.lower()]
james_prentice_name = [name for name in names_enron_data if 'james' in name.lower() and 'prentice' in name.lower()]
# print(f"Stock features: {stock_features}")
# print(f"Name of the person: {james_prentice_name}")

# Print the total value of the stock belonging to James Prentice
print(f"Total stock value of James Prentice: {enron_data[james_prentice_name[0]][stock_features[0]]}")


# Print the total value of the stock belonging to Wesley Colwell
wesley_colwell_name = [name for name in names_enron_data if 'wesley' in name.lower() and 'colwell' in name.lower()]
# print(f"Name of the person: {wesley_colwell_name}")
# Print Number of emails sent from Wesley Colwell to POIs
print(f"Total Number of emails of Wesley Colwell: {enron_data[wesley_colwell_name[0]]['from_this_person_to_poi']}")


# Print the  value of stock options belonging to Jeffrey K Skillin
jeffrey_skillin_name = [name for name in names_enron_data if 'jeffrey' in name.lower() and 'skillin' in name.lower()]
# print(f"Name of the person: {wesley_colwell_name}")

# Print the value of stock options belonging to Jeffrey K Skillin
print(f"The value of stock options belonging to Jeffrey K Skillin: {enron_data[jeffrey_skillin_name[0]]['exercised_stock_options']}")

# Print the  value of total payments to Lay, Skilling and Fastow
# Lay_Skilling_Fastow_names = [name for name in names_enron_data if 'jeffrey' in name.lower() or 'lay' in name.lower() or 'fastow' in name.lower()]
Lay_Skilling_Fastow_names = ['LAY KENNETH L', 'FASTOW ANDREW S', 'SKILLING JEFFREY K']
print(f"Name of the person: {Lay_Skilling_Fastow_names}")
total_payments = {name: enron_data[name]['total_payments'] for name in Lay_Skilling_Fastow_names}
# Print the name of the person with the highest total payments
max_total_payments = max(total_payments, key=total_payments.get)
print(f"Name of the person with the highest total payments: {max_total_payments}")
