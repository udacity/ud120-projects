#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from sklearn.metrics import f1_score
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'bonus', "long_term_incentive", "from_this_person_to_poi", "from_poi_to_this_person", "shared_receipt_with_poi", "to_messages", "from_messages", "other", "expenses", "exercised_stock_options", "restricted_stock", "total_payments", "total_stock_value"]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
for key in data_dict:
    data = data_dict[key]
    from_messages = data["from_messages"]
    to_messages = data["to_messages"]
    from_this_person_to_poi = data["from_this_person_to_poi"]
    from_poi_to_this_person = data["from_poi_to_this_person"]
    if (from_messages == 'NaN' or  to_messages == 'NaN' or from_this_person_to_poi == 'NaN' or from_poi_to_this_person == 'NaN'):
        data["ratio_from_to_poi"] = 0
        data["ratio_to_from_poi"] = 0
    else:
        data["ratio_from_to_poi"] = float(from_this_person_to_poi) / from_messages
        data["ratio_to_from_poi"] = float(from_poi_to_this_person) / to_messages

my_dataset = data_dict
data_dict.pop("TOTAL", 0)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Thank to http://luizschiller.com/enron/ for information about pipeline and scaling

SCALER = [None, preprocessing.StandardScaler()]
SELECTOR__K = [10, 'all']
REDUCER__N_COMPONENTS = [2, 4, 6, 8, 10]

default_param_grid = {
    'scaler': SCALER,
    'selector__k': SELECTOR__K,
    'reducer__n_components': REDUCER__N_COMPONENTS
}

default_pipline = [
        ('scaler', preprocessing.StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42))
]

classifiers = {
    'navie_bays': {
        'classifier': GaussianNB()
    },
    'decision_tree': {
        'classifier': DecisionTreeClassifier(),
        'parameters': {
            'classifier__criterion': ['gini'],
            'classifier__splitter': ['random'],
            'classifier__min_samples_split': [2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 50],
            'classifier__class_weight': ['balanced', None]
        }
    },
    'support_vector_machine': {
        'classifier': SVC(kernel='rbf', class_weight='balanced'),
        'parameters': {
            'classifier__C': [1e3, 5e3, 1e4, 5e4, 1e5],
            'classifier__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        }
    }
}


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# PARAMS scoring selects the f1 scoring algo
# cv : int, cross-validation generator or an iterable, optional
# f1 evaluating based on  recall and precision.
def fit_and_score(key, pipeline, param_grid):
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(pipe, param_grid=param_grid,  scoring='f1', cv=sss, verbose=0)
    grid_search.fit(np.array(features_train), np.array(labels_train))
    pred = grid_search.best_estimator_.predict(features_test)
    score = f1_score(labels_test, pred)
    print "f1 score for classifier ### ", key, ' #### - ', score
    print "best parameter------------------"
    print  grid_search.best_params_
    print "-----------------"

    return (grid_search.best_estimator_, score)


best_classifier_score = 0
best_classifier = 0
for classifierKey in classifiers:
    classifierData = classifiers[classifierKey]
    classifier = classifierData['classifier']

    parameters = classifierData.get('parameters')

    pipelineData = default_pipline[:]
    pipelineData.append(('classifier', classifier))
    pipe = Pipeline(pipelineData)
    param = dict(default_param_grid)
    if (parameters != None):
        param.update(parameters)

    (clf, score) = fit_and_score(classifierKey, pipe, param)
    if (score > best_classifier_score):
        best_classifier = clf


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(best_classifier, my_dataset, features_list)
