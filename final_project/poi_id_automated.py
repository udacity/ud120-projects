#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

### helper functions for data wrangling
from data_wrangling import *

### helper functions for Machine Learning
from data_learning import *

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### features_list = ['poi','salary'] # You will need to use more 

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Target label to identify whether a person is poi or not
target_label = 'poi'

### Now get a comprehensive list of features from the data excluding 
### string features and first feature as poi which is the label
total_features_list = get_features(data_dict, target_label)

### Add valid counts and total counts for inspection
features_specs = count_values(data_dict, total_features_list)

### Task 2: Remove outliers

### List of persons
persons_list = data_dict.keys()

### from manual inspection of the above list of persons I decided to choose
### 2 person TOTAL and THE TRAVEL AGENCY IN THE PARK as my outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
data_dict = remove_outlier(data_dict, outliers)

### Task 3: Create new feature(s)

### Upon manual inspection of the features_list I observed that
### no feature related to their financial status such as stocks
data_dict, features_list = add_features_totals(data_dict, total_features_list)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Splits data dictionary into features dataframe and labels dataframe.
labels, features = targetFeatureSplitPandas(my_dataset)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### StratifiedShuffleSplits for 100 internal cross-validation splits
### within the grid-search.
sss = StratifiedShuffleSplit(labels, n_iter=100)

### Since we need to optimize our parameters for both recall and precision
### better than 0.3, so I have chosen F1 as my scoring metric for GridSearchCV
scoring_metric = 'f1'

### Logistic Regression Classifier Pipeline

print "Logistic Regression Classifier starts...................\n"

### Splits data dictionary into features dataframe and labels dataframe.
labels, features = targetFeatureSplitPandas(my_dataset)

### Get pipeline and parameters for GridSearchCV for above classifier.
### If you want to run the optimization again then change reoptimize to False
### Please be aware, it might take significant time.
pipe, parameters = get_logReg_optimizer(reoptimize = False)

### Find the optimized parameters using GridSearchCV
grid_searcher = GridSearchCV(pipe, param_grid=parameters, cv=sss, 
                             scoring = scoring_metric)
grid_searcher.fit(features, labels)

### Assign the best estimator to final LR classifier
lr_clf = grid_searcher.best_estimator_

### combine labels and features into data dictionary for logistic regression
lr_dataset = combineLabelsFeatures(labels, features)


test_classifier(lr_clf, lr_dataset, features_list)

print "Logistic Regression Classifier ends...................\n"

### K Neighbor Classifier Pipeline

print "K Neighbor Classifier starts...................\n"

### Splits data dictionary into features dataframe and labels dataframe.
labels, features = targetFeatureSplitPandas(my_dataset)

### Get pipeline and parameters for GridSearchCV for K Neighbor classifier.
### If you want to run the optimization again then change reoptimize to False
### Please be aware, it might take significant time.
pipe, parameters = get_kNeighbor_optimizer(reoptimize = False)

### Find the optimized parameters using GridSearchCV
grid_searcher = GridSearchCV(pipe, param_grid=parameters, cv=sss,
                             scoring = scoring_metric)
grid_searcher.fit(features, labels)

### Assign the best estimator to final LR classifier
kn_clf = grid_searcher.best_estimator_

### combine labels and features into data dictionary for logistic regression
kn_dataset = combineLabelsFeatures(labels, features)

test_classifier(kn_clf, kn_dataset, features_list)

print "K Neighbor Classifier ends...................\n"

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

my_dataset = lr_dataset
clf = lr_clf
    
#test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
#dump_classifier_and_data(clf, my_dataset, features_list)
