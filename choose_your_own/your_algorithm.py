#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import make_terrain_data
from class_vis import pretty_picture
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def make_prediction():
    features_train, labels_train, features_test, labels_test = make_terrain_data()

    # the training data (features_train, labels_train) have both "fast" and "slow"
    # points mixed together--separate them so we can give them different colors
    # in the scatterplot and identify them visually

    grade_fast = [
        features_train[ii][0]
        for ii in range(0, len(features_train))
        if labels_train[ii] == 0
    ]

    bumpy_fast = [
        features_train[ii][1]
        for ii in range(0, len(features_train))
        if labels_train[ii] == 0
    ]

    grade_slow = [
        features_train[ii][0]
        for ii in range(0, len(features_train))
        if labels_train[ii] == 1
    ]

    bumpy_slow = [
        features_train[ii][1]
        for ii in range(0, len(features_train))
        if labels_train[ii] == 1
    ]

    # initial visualization
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
    plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.show()

    clf = RandomForestClassifier(min_samples_split=40)
    # clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(features_train, labels_train)
    prediction = clf.predict(features_test)

    print("accuracy: {}".format(accuracy_score(labels_test, prediction)))

    try:
        pretty_picture(clf, features_test, labels_test)
    except NameError:
        pass


if __name__ == "__main__":
    make_prediction()
