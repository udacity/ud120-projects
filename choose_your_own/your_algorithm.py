#!/usr/bin/python

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
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
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# Initialize random forest classifier
clf = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=50,
    min_samples_split=25,
    max_features=None,
    random_state=0,
    bootstrap=True
)

# Fit classifier to given test data
clf.fit(features_train, labels_train)

# Predict output using trained model
labels_pred = clf.predict(features_test)

# Calculate accuracy
acc = accuracy_score(labels_test, labels_pred)

# Print the classification graphic
try:
    prettyPicture(clf, features_test, labels_test, accuracy=acc)
except NameError:
    print "Could not find a classifier."
