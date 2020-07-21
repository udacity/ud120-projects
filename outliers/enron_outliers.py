#!/usr/bin/python

import os
import pickle
import matplotlib.pyplot as plt
from tools.feature_format import feature_format, target_feature_split


BASE_PATH = os.path.split(os.path.abspath(os.getcwd()))[0]


def clean_outliers():
    # read in data dictionary, convert to numpy array
    data_dict = pickle.load(open(os.path.join(BASE_PATH, "final_project/final_project_dataset.pkl"), "r"))
    del data_dict["TOTAL"]
    features = ["salary", "bonus"]
    data = feature_format(data_dict, features)

    for point in data:
        salary = point[0]
        bonus = point[1]
        plt.scatter(salary, bonus)

    plt.xlabel("salary")
    plt.ylabel("bonus")
    plt.show()


if __name__ == "__main__":
    clean_outliers()
