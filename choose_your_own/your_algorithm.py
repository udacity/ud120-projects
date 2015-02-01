#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
#################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
from time import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Classifier list
classifers_dict = {'AdaBoost': AdaBoostClassifier(n_estimators=100,
                                                  learning_rate=1),
                    'Random Forest': RandomForestClassifier(n_estimators=10,
                                                            # max_depth=None,
                                                            min_samples_split=50,
                                                            random_state=0),
                    'Naive Bayes': GaussianNB(),
                    'SVM': SVC(C=1, kernel='poly', degree=2, gamma=0.),
                    'Tree': DecisionTreeClassifier(),
                    'KNN': KNeighborsClassifier(n_neighbors=20, weights='distance', p=3)
                    }


for name, clf in classifers_dict.iteritems():
    print name
    t0 = time()
    clf.fit(features_train, labels_train)
    print "\tTime to train: ", time() - t0, "s"

    pred = clf.predict(features_test)
    acc = accuracy_score(labels_test, pred)
    print "\tPrediction accuracy: ", acc



try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
