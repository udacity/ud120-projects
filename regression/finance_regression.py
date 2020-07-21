#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tools.feature_format import feature_format, target_feature_split


BASE_PATH = os.path.split(os.path.abspath(os.getcwd()))[0]

dictionary = pickle.load(open(os.path.join(BASE_PATH, "final_project/final_project_dataset_modified.pkl"), "r"))


def predict():
    # list the features you want to look at--first item in the
    # list will be the "target" feature
    features_list = ["bonus", "salary"]
    data = feature_format(dictionary, features_list, remove_any_zeroes=True)
    target, features = target_feature_split(data)

    # training-testing split needed in regression, just like classification
    feature_train, feature_test, target_train, target_test = train_test_split(
        features, target, test_size=0.5, random_state=42
    )
    train_color = "b"
    test_color = "r"

    reg = LinearRegression()
    reg.fit(feature_train, target_train)
    prediction = reg.predict(feature_test)

    print("slope: {}".format(reg.coef_))
    print("intercept: {}".format(reg.intercept_))
    print("r2 score: {}".format(r2_score(target_test, prediction)))
    print("\n")

    # draw the scatterplot, with color-coded training and testing points

    for feature, target in zip(feature_test, target_test):
        plt.scatter(feature, target, color=test_color)
    for feature, target in zip(feature_train, target_train):
        plt.scatter(feature, target, color=train_color)

    # labels for the legend
    plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
    plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

    # draw the regression line, once it's coded
    try:
        plt.plot(feature_test, prediction)
    except NameError:
        pass

    reg.fit(feature_test, target_test)
    prediction = reg.predict(feature_train)
    plt.plot(feature_train, prediction, color="b")
    print("slope: {}".format(reg.coef_))
    print("intercept: {}".format(reg.intercept_))
    print("r2 score: {}".format(r2_score(target_train, prediction)))

    plt.xlabel(features_list[1])
    plt.ylabel(features_list[0])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    predict()
