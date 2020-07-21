#!/usr/bin/python

import random
import numpy
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from outlier_cleaner import outlier_cleaner


BASE_PATH = os.path.split(os.path.abspath(os.getcwd()))[0]


def outlier_removal():
    # load up some practice data with outliers in it
    ages = pickle.load(
        open(os.path.join(BASE_PATH, "outliers/practice_outliers_ages.pkl"), "r")
    )
    net_worths = pickle.load(
        open(os.path.join(BASE_PATH, "outliers/practice_outliers_net_worths.pkl"), "r")
    )

    # ages and net_worths need to be reshaped into 2D numpy arrays
    # second argument of reshape command is a tuple of integers: (n_rows, n_columns)
    # by convention, n_rows is the number of data points
    # and n_columns is the number of features
    ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

    ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(
        ages, net_worths, test_size=0.1, random_state=42
    )

    reg = LinearRegression()
    reg.fit(ages_train, net_worths_train)
    predictions = reg.predict(ages_test)
    print("slope before cleaning: {}".format(reg.coef_))
    print("intercept before cleaning: {}".format(reg.intercept_))
    print("r2 score before cleaning: {}".format(r2_score(net_worths_test, predictions)))
    print("\n")

    try:
        plt.plot(ages_test, predictions, color="blue")
    except NameError:
        pass
    plt.scatter(ages, net_worths)
    plt.show()

    # identify and remove the most outlier-y points
    cleaned_data = []
    try:
        predictions = reg.predict(ages_train)
        cleaned_data = outlier_cleaner(predictions, ages_train, net_worths_train)
    except NameError:
        print("your regression object doesn't exist, or isn't name reg")
        print("can't make predictions to use in identifying outliers")

    # only run this code if cleaned_data is returning data
    if len(cleaned_data) > 0:
        ages, net_worths, errors = zip(*cleaned_data)
        ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
        net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

        # refit your cleaned data!
        try:
            reg.fit(ages, net_worths)
            predictions = reg.predict(ages_test)
            plt.plot(ages_test, predictions, color="blue")

            print("slope: {}".format(reg.coef_))
            print("intercept: {}".format(reg.intercept_))
            print("r2 score: {}".format(r2_score(net_worths_test, predictions)))
        except NameError:
            print("you don't seem to have regression imported/created,")
            print("   or else your regression object isn't named reg")
            print("   either way, only draw the scatter plot of the cleaned data")
        plt.scatter(ages, net_worths)
        plt.xlabel("ages")
        plt.ylabel("net worths")
        plt.show()
    else:
        print("outlier_cleaner() is returning an empty list, no refitting to be done")


if __name__ == "__main__":
    outlier_removal()
