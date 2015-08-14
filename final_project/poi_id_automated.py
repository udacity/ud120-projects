#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

### Classifier
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition

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

### Extract features and labels from dataset
labels, features = targetFeatureSplitPandas(my_dataset)

X_features = list(features.columns)
features_list = ['poi'] + X_features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

sk_fold = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1)

pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                             ('selection', SelectKBest(score_func=f_classif)),
                             ('reducer', PCA()),
                             ('classifier', LogisticRegression())
                             ])
                             
params = {'selection__k': [10, 15, 17, 19],
          'classifier__C': [1e-5, 1e-2, 1e-1, 1, 10, 100],
          'classifier__class_weight': [{True: 12, False: 1},
                                       {True: 10, False: 1},
                                      {True: 8, False: 1},
                                      {True: 15, False: 1},
                                      {True: 20, False: 1},
                                      'auto', None],
          'classifier__tol': [1e-1, 1e-2, 1e-4, 1e-8, 1e-16,
                              1e-32, 1e-64, 1e-128, 1e-256],
          'reducer__n_components': [1, 2, 3, 4, 5, .25, .4, .5, .6,
                                    .75, .9, 'mle'],
          'reducer__whiten': [True, False]
          }

scoring_metric = 'recall'

grid_searcher = GridSearchCV(pipeline, param_grid=params, cv=sk_fold, 
                             n_jobs=-1, scoring=scoring_metric, verbose=0)
                             
grid_searcher.fit(features, y=labels)

mask = grid_searcher.best_estimator_.named_steps['selection'].get_support()
top_features = [x for (x, boolean) in zip(features, mask) if boolean]
n_pca_components = grid_searcher.best_estimator_.named_steps['reducer'].n_components_

print "Cross-validated {0} score: {1}".format(scoring_metric, grid_searcher.best_score_)
print "{0} features selected".format(len(top_features))
print "Reduced to {0} PCA components".format(n_pca_components)
###################
# Print the parameters used in the model selected from grid search
print "Params: ", grid_searcher.best_params_ 
###################

clf = grid_searcher.best_estimator_

### combine labels and features into data dictionary
my_dataset = combineLabelsFeatures(labels, features)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Extract features and labels frouum dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
    
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, features_list)