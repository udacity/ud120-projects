#!/usr/bin/python
"""Skeleton code for k-means clustering mini-project."""

import sys
import pickle
import matplotlib.pyplot as plt

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


def draw(pred, features, poi, mark_poi=False, name="image.png",
         f1_name="feature 1", f2_name="feature 2"):
    """Some plotting code designed to help you visualize your clusters."""
    # Plot each cluster with a different color--add more colors for
    # drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    # If you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(
                    features[ii][0], features[ii][1], color="r", marker="*")

    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

# Load the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load(
    open("../final_project/final_project_dataset.pkl", "r"))

# There's an outlier--remove it!
data_dict.pop("TOTAL", 0)

# The input features we want to use
# Can be any key in the person-level dictionary (salary, director_fees, etc.)
features_studied = ["salary", "exercised_stock_options"]
features_list = ["poi"] + features_studied
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)

for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()

# Cluster here; create predictions of the cluster labels
# for the data and store them to a list called pred

# Rename the "name" parameter when you change the number of features
# so that the figure gets saved to a different file
try:
    draw(pred, finance_features, poi,
         mark_poi=False, name="clusters.pdf",
         f1_name="salary", f2_name="exercised_stock_options")

except NameError:
    print "no predictions object named pred found, no clusters to plot"
