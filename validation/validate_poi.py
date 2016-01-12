#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


### it's all yours from here forward!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.30, random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print 'Accuracy score = ', accuracy_score(labels_test, pred)

print 'Predicted number of POI = ', sum(pred)
print 'Size of test set = ', len(pred)

# Assuming no POI identified
fake_pred = [0] * len(pred)
print 'Fake Accuracy = ', accuracy_score(labels_test, fake_pred)

# How many true positives
true_positives = 0
for index in range(0, len(pred)):
    if (int(pred[index]) & int(labels_test[index])):
        true_positives +=  1
print 'Number of True Positives = ', true_positives

# Precision
from sklearn.metrics import precision_score
print 'Precision score = ', precision_score(labels_test, pred)

# Recall
from sklearn.metrics import recall_score
print 'Recall score = ', recall_score(labels_test, pred)

# Made-up predictions
# predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
# true labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# tp = 6
# tn = 9
# fn = 2
# fp = 3
