#!/usr/bin/python

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

np.random.seed(9)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]


################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

clf = KNeighborsClassifier(n_jobs=-1)

features_train = np.array(features_train)
features_test = np.array(features_test)

logger.info("Training model...")
t = time.time()

space = {
    "n_neighbors": [3, 5, 10, 25],
    "weights": ["uniform", "distance"],
    "leaf_size": [2, 10, 30, 100],
    "p": [1, 2, 10],
}

clf = GridSearchCV(clf, space, n_jobs=-1, refit=True)
clf.fit(features_train, labels_train)

logger.info(f"\nBest parameters: {clf.best_params_}\n")

logger.info("Done training. Took {:.3} minutes".format((time.time() - t) / 60))

y_pred = clf.predict(features_test)
acc = accuracy_score(labels_test, y_pred)
logger.info("Accuracy: {:.3}".format(acc))


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
prettyPicture(clf, features_test, labels_test)
plt.show()
