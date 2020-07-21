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

import os
import pickle


BASE_PATH = os.path.split(os.path.abspath(os.getcwd()))[0]


def count_missing_payments(data, poi=False):
    count = 0
    for key, value in data.items():
        if value["total_payments"] == "NaN":
            count += 1

            if poi and value["poi"] == 0:
                count -= 1
    return count


def count_known_emails(data):
    count = 0
    for key, value in data.items():
        if value["email_address"] != "NaN":
            count += 1
    return count


def count_quantified_salaries(data):
    count = 0
    for key, value in data.items():
        if value["salary"] != "NaN":
            count += 1
    return count


def count_pois(data):
    count = 0
    for key, value in data.items():
        if value["poi"] == 1:
            count += 1
    return count


def get_person(data, person):
    return data.get(person.upper())


def load_enron_date():
    return pickle.load(
        open(os.path.join(BASE_PATH, "final_project/final_project_dataset.pkl"), "rb")
    )


if __name__ == "__main__":
    enron_data = load_enron_date()
    print(enron_data)
    number_of_data_points = len(enron_data.keys())
    print("no. of data points: {}".format(number_of_data_points))
    print(
        "no. of features per data point: {}".format(
            len(enron_data.get(next(iter(enron_data))).keys())
        )
    )
    print(
        "available features: {}".format(enron_data.get(next(iter(enron_data))).keys())
    )
    print("no. of POIs: {}".format(count_pois(enron_data)))
    print(
        "total value of stock for James Prentice: {}".format(
            get_person(enron_data, "PRENTICE JAMES").get("total_stock_value")
        )
    )
    print(
        "emails to POIs by Wesley Colwell: {}".format(
            get_person(enron_data, "COLWELL WESLEY").get("from_this_person_to_poi")
        )
    )
    print(
        "value of stock options exercised by Jeff Skilling: {}".format(
            get_person(enron_data, "SKILLING JEFFREY K").get("exercised_stock_options")
        )
    )
    print(
        "no. of people with quantified salaries: {}".format(
            count_quantified_salaries(enron_data)
        )
    )
    print(
        "no. of people with known email addresses: {}".format(
            count_known_emails(enron_data)
        )
    )

    people_with_missing_total_payments = count_missing_payments(enron_data)
    print(
        "no. of people with NaN for total payments: {}".format(
            people_with_missing_total_payments
        )
    )
    print(
        "no. of people with NaN for total payments as a percentage: {}".format(
            (float(people_with_missing_total_payments) / float(number_of_data_points)) * 100
        )
    )

    poi_with_missing_total_payments = count_missing_payments(enron_data, poi=True)
    print(
        "no. of POI with NaN for total payments: {}".format(
            poi_with_missing_total_payments
        )
    )
    print(
        "no. of POI with NaN for total payments as a percentage: {}".format(
            (float(poi_with_missing_total_payments) / float(number_of_data_points)) * 100
        )
    )
