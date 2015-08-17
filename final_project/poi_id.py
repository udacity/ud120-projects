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

### Store to temp_dataset for easy export below.
temp_dataset = data_dict

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

################ Logistic Regression Classifier Pipeline starts ################

### Splits data dictionary into features dataframe and labels dataframe.
labels, features = targetFeatureSplitPandas(temp_dataset)

### StratifiedShuffleSplits for 100 internal cross-validation splits
### within the grid-search.
sss = StratifiedShuffleSplit(labels, n_iter=1000)

### Get pipeline and parameters for GridSearchCV for above classifier.
### If you want to run the optimization again then change reoptimize to False
### Please be aware, it might take significant time.
pipe, parameters, scoring_metric = get_logReg_optimizer(reoptimize = False)

### Find the optimized parameters using GridSearchCV
grid_searcher_lr = GridSearchCV(pipe, param_grid=parameters, cv=sss, 
                             scoring = scoring_metric)
grid_searcher_lr.fit(features, labels)

### Score of best_estimator on the left out data
#print "best score is {0}".format(grid_searcher_rf.best_score_)

### Assign the best estimator to final LR classifier
lr_clf = grid_searcher_lr.best_estimator_

### combine labels and features into data dictionary for logistic regression
lr_dataset = combineLabelsFeatures(labels, features)

#test_classifier(lr_clf, lr_dataset, features_list)

################# Logistic Regression Classifier Pipeline ends #################

#################### K Neighbors Classifier Pipeline starts ####################

### Splits data dictionary into features dataframe and labels dataframe.
labels, features = targetFeatureSplitPandas(temp_dataset)

### StratifiedShuffleSplits for 100 internal cross-validation splits
### within the grid-search.
sss = StratifiedShuffleSplit(labels, n_iter=1000)

### Get pipeline and parameters for GridSearchCV for K Neighbors classifier.
### If you want to run the optimization again then change reoptimize to False
### Please be aware, it might take significant time.
pipe, parameters, scoring_metric = get_kNeighbor_optimizer(reoptimize = False)

### Find the optimized parameters using GridSearchCV
grid_searcher_kn = GridSearchCV(pipe, param_grid=parameters, cv=sss,
                             scoring = scoring_metric)
grid_searcher_kn.fit(features, labels)

### Score of best_estimator on the left out data
#print "best score is {0}".format(grid_searcher_kn.best_score_)

### Assign the best estimator to final K Neighbors classifier
kn_clf = grid_searcher_kn.best_estimator_

### combine labels and features into data dictionary for K Neighbors
kn_dataset = combineLabelsFeatures(labels, features)

#test_classifier(kn_clf, kn_dataset, features_list)

##################### K Neighbors Classifier Pipeline ends #####################

#################### Support Vector Machine Pipeline starts ####################

### Splits data dictionary into features dataframe and labels dataframe.
labels, features = targetFeatureSplitPandas(temp_dataset)

### StratifiedShuffleSplits for 100 internal cross-validation splits
### within the grid-search.
sss = StratifiedShuffleSplit(labels, n_iter=1000)

### Get pipeline and parameters for GridSearchCV for SVM.
### If you want to run the optimization again then change reoptimize to False
### Please be aware, it might take significant time.
pipe, parameters, scoring_metric = get_svm_optimizer(reoptimize = False)

### Find the optimized parameters using GridSearchCV
grid_searcher_svm = GridSearchCV(pipe, param_grid=parameters, cv=sss,
                             scoring = scoring_metric)
grid_searcher_svm.fit(features, labels)

### Score of best_estimator on the left out data
#print "best score is {0}".format(grid_searcher_svm.best_score_)

### Assign the best estimator to final SVM classifier
svm_clf = grid_searcher_svm.best_estimator_

### combine labels and features into data dictionary for SVM
svm_dataset = combineLabelsFeatures(labels, features)

#test_classifier(svm_clf, svm_dataset, features_list)

##################### Support Vector Machine Pipeline ends #####################

######################## Random Forest Pipeline starts #########################

### Splits data dictionary into features dataframe and labels dataframe.
labels, features = targetFeatureSplitPandas(temp_dataset)

### StratifiedShuffleSplits for 100 internal cross-validation splits
### within the grid-search.
sss = StratifiedShuffleSplit(labels, n_iter=1000)

### Get pipeline and parameters for GridSearchCV for K Neighbor classifier.
### If you want to run the optimization again then change reoptimize to False
### Please be aware, it might take significant time.
pipe, parameters, scoring_metric = get_rForest_optimizer(reoptimize = False)

### Find the optimized parameters using GridSearchCV
grid_searcher_rf = GridSearchCV(pipe, param_grid=parameters, cv=sss,
                             scoring = scoring_metric)
grid_searcher_rf.fit(features, labels)

### Score of best_estimator on the left out data
#print "best score is {0}".format(grid_searcher_rf.best_score_)

### Assign the best estimator to final Random Forest classifier
rf_clf = grid_searcher_rf.best_estimator_

### combine labels and features into data dictionary for logistic regression
rf_dataset = combineLabelsFeatures(labels, features)

#test_classifier(rf_clf, rf_dataset, features_list)

######################### Random Forest Pipeline ends ##########################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Based upon the best score from GridSearchCV pipeline, I have decided to
### chosse Support Vector Machine as my final algorithm
my_dataset = svm_dataset
clf = svm_clf

mask = clf.named_steps['select'].get_support()
top_features = [x for (x, boolean) in zip(features, mask) if boolean]
n_pca_components = clf.named_steps['pca'].n_components_
    
print "{0} best features were selected".format(len(top_features))
print "Reduced to {0} PCA components".format(n_pca_components)
    
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, features_list)
