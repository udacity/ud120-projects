#!/usr/bin/python
import sys
import pickle
from pprint import pprint
from time import time

from sklearn import svm, linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from helper import _remove_outlier, _scale_data, _get_classifier, _cross_validate, _get_train_test_data, \
    _select_features, _get_parameters, _evaluate_grid_search, _get_features, _get_new_features, _get_new_classifier

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

"""
TO DO:
- select features
    - PCA
    - univariate feature selection
- remove outliers
- create new features
- try different classifiers: 
    - SVM, Naive Bayes, Decision Tree, Ada Boost 
    - Does the classifier require scaling (sklearn.preprocessing)?
- validation:
    - cross-validation, k-fold
    - model evaluation
    - validation curve, learning curve
- grid search:
>>>inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
>>>svm = SVC(kernel="rbf")
>>>p_grid = {"C": [1, 10, 100],"gamma": [.01, .1]}
GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
- use pipelines
>>> estimators = [('reduce_dim', PCA()), ('clf', SVC())]
>>> pipe = Pipeline(estimators)
"""

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = _get_features()
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "br") as data_file:
    data_dict = pickle.load(data_file)
print('The number of person is {0}.'.format(len(data_dict.keys())))
### Task 2: Remove outliers
data_dict = _remove_outlier(data_dict)
### Task 3: Create new feature(s)
### Extract features and labels from dataset for local testing
# data: array of features per person
data = featureFormat(data_dict, features_list, sort_keys = True)
# labels:  1 for poi, 0 for non-poi, features: np.array([])
labels, features = targetFeatureSplit(data)
features = _scale_data(features)
print('The number of person after feature formatting is {0}.'.format(len(labels)))
feature_train, feature_test, label_train, label_test = _get_train_test_data(features, labels)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
feat_select = _select_features('k_best')
dim_reduct = _select_features('pca')
clf = _get_classifier('svm')
#print('Components: {0}'.format(pca.components_))
#print('Explained variance: {0}'.format(pca.explained_variance_))
#print('Explained variance ratio: {0}'.format(pca.explained_variance_ratio_ ))
#print('Number of features: {0}'.format(len(myfeatures)))
#print('Number of components: {0}'.format(pca.n_components_))
#mypipeline = Pipeline([('feat_select', feat_select), ('dim_reduct', dim_reduct), ('clf', clf)])
mypipeline = Pipeline([('feat_select', feat_select),  ('clf', clf)])
# grid search
parameters = _get_parameters()
scoring = ['precision', 'recall']
grid_search = GridSearchCV(mypipeline, parameters, scoring=scoring)
_evaluate_grid_search(grid_search, mypipeline, parameters, feature_train, label_train)
# this is for fixed parameters
#mypipeline.set_params(feat_select__k=5, dim_reduct__n_components=3, clf__C=1e2, clf__gamma=1, clf__kernel='rbf').fit(feature_train, label_train)
mypipeline.set_params(feat_select__k=10, clf__C=1e3, clf__gamma=1e-1, clf__kernel='linear').fit(feature_train, label_train)
new_features =_get_new_features(mypipeline)
new_clf = _get_new_classifier(mypipeline)
_cross_validate(mypipeline, feature_train, label_train)
"""
Why is the test_recall = 0 and test_precision = 0 and accuracy = 0.8?  
"""
prediction = mypipeline.predict(feature_test)

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.model_selection import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(new_clf, data_dict, new_features)
