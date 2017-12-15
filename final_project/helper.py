from pprint import pprint
from time import time
import sys
import itertools
import numpy as np
from sklearn import svm, linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB

sys.path.append("../tools/")
from tester import test_classifier


def _get_features():
    features_list = ['poi',
                     'bonus',
                     'deferral_payments',
                     'deferred_income',
                     'director_fees',
                     'email_address',
                     'exercised_stock_options',
                     'expenses',
                     'from_messages',
                     'from_poi_to_this_person',
                     'from_this_person_to_poi',
                     'loan_advances',
                     'long_term_incentive',
                     'other',
                     'restricted_stock',
                     'restricted_stock_deferred',
                     'salary',
                     'shared_receipt_with_poi',
                     'to_messages',
                     'total_payments',
                     'total_stock_value']

    features_list.remove('email_address')
    myfeatures = ['poi',
                  'from_poi_to_this_person',
                  'to_messages',
                  'from_this_person_to_poi',
                  'from_messages',
                  'salary',
                  'bonus',
                  'deferral_payments',
                  'deferred_income',
                  'director_fees',
                  'expenses',
                  'loan_advances',
                  'long_term_incentive',
                  'total_payments']
    return myfeatures

def _create_new_features(features):
    """
    features[0]: 'from_poi_to_this_person',
    features[1]: 'to_messages',
    features[2]: 'from_this_person_to_poi',
    features[3]: 'from_messages',
    :param features:
    :return: new features, i.e. ratio of emails from/to poi and total from/to emails
    """
    for vector, i in zip(features, range(len(features))):
        if vector[0] != 0 and vector[3] != 0:
            vector[0] = vector[0]/vector[1]
            vector[1] = vector[2]/vector[3]
        else:
            vector[0] = 0
            vector[1] = 0
        features[i] = vector
    new_features = np.delete(np.array(features), [2, 3], axis=1)
    return new_features

def _remove_outlier(data_dict):
    outlier_list = ['TOTAL', 'LAY KENNETH L', 'SKILLING JEFFREY K']
    for name in list(data_dict.keys()):
        if name in outlier_list:
            data_dict.pop(name)
    return data_dict

def _scale_data(features):
    #features_scaled = np.array([feature/feature.max() for feature in features])
    #return features_scaled
    myfeatures = []
    for feature in features:
        array_min = feature.min()
        array_max = feature.max()
        feature = (feature-array_min)/(array_max-array_min)
        myfeatures.append(feature)
    return np.array(myfeatures)

def _select_features(key):
    feat_select = {'pca': PCA(n_components=5),
                   'var_threshold': VarianceThreshold(threshold = 0.1),
                   'k_best': SelectKBest(k=5)}
    return feat_select[key]

def _get_classifier(key):
    clf_dict ={'svm': svm.SVC(kernel='rbf'),
               'ada_boost': AdaBoostClassifier(),
               'nb': GaussianNB(),
               'lin_reg': linear_model.LinearRegression()}
    return clf_dict[key]

def _get_train_test_data(features, labels):
    feature_train, feature_test, label_train, label_test = train_test_split(
    features, labels, test_size = 0.4, random_state = 0)
    return feature_train, feature_test, label_train, label_test


def _cross_validate(pipeline, features, labels):
    """
     precision = Tp/(Tp + Fp)
     recall = Tp/(Tp + Fn)
    """
    print('My pipeline: {0}'.format(pipeline))
    scoring = ['precision', 'recall', 'accuracy']
    sss = StratifiedShuffleSplit(n_splits=50, test_size=0.25, random_state=42)
    scores = cross_validate(estimator=pipeline,
                            X=features,
                            y=labels,
                            scoring=scoring,
                            verbose=1,
                            cv=sss,
                            return_train_score='warn')
    print(scores.keys())
    train_recall = _get_mean_and_std(scores['train_recall'])
    test_recall = _get_mean_and_std(scores['test_recall'])
    train_precision = _get_mean_and_std(scores['train_precision'])
    test_precision = _get_mean_and_std(scores['test_precision'])
    accuracy = _get_mean_and_std(scores['test_accuracy'])
    print('train_recall: {0:0.3f} +/- {1:0.3f}'.format(train_recall[0], train_recall[1]))
    print('test_recall: {0:0.3f} +/- {1:0.3f}'.format(test_recall[0], test_recall[1]))
    print('train_precision: {0:0.3f} +/- {1:0.3f}'.format(train_precision[0], train_precision[1]))
    print('test_precision: {0:0.3f} +/- {1:0.3f}'.format(test_precision[0], test_precision[1]))
    print('accuracy: {0:0.3f} +/- {1:0.3f}'.format(accuracy[0], accuracy[1]))


def _get_mean_and_std(array):
    mean = array.mean()
    std = array.std()
    return mean, std

def _get_parameters():
    parameters = {#'feat_select__k': (18, 19),
                  'dim_reduct__n_components': (3, 5, 7, 9, 11),
                  'clf__kernel': ('rbf',),
                  'clf__C': (1e2, 5e2, 1e3),
                  'clf__gamma': (0.5e-1, 1e-1, 5e-1)
                  }
    return parameters

def _get_new_features(pipeline):
    '''
    gets either the 'dim_reduct' or 'feat_select' step from the pipeline
    after optimization
    :param pipeline:
    :return: PCA components if step if 'dim_reduct',
            selected features if step is 'feat_select'
    '''
    step = pipeline.named_steps.get('feat_select')
    if step:
        ind_selected_feat = step.get_support(indices=True)
        print('Indices of selected features: {0}'.format(ind_selected_feat))
        myfeatures = np.array(_get_features()[1:])
        selected_feat = myfeatures[ind_selected_feat]
        print('selected features: {0}'.format(selected_feat))
        return np.insert(selected_feat, 0, 'poi')

def _get_new_classifier(pipeline):
    '''

    :param pipeline:
    :return: new classifier after the optimization
    '''
    clf = pipeline.named_steps.get('clf')
    print('clf: {0}'.format(clf))
    return clf


def _evaluate_grid_search(grid_search, mypipeline, parameters, feature, label, scoring):
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in mypipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(feature, label)
    print("done in {0:.3f} s".format(time() - t0))
    print(grid_search.cv_results_)
    print("Scorer: {0}".format(grid_search.scorer_))
    print("Best score: {0:.3f}".format(grid_search.best_score_))
    print("Best estimator: {0}".format(grid_search.best_estimator_))
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

def _test_pipeline(pipeline, params, feature_train, label_train, data_dict, features_list, folds):
    params_names = params.keys()
    params_values = list(params.values())
    params_values_product = list(itertools.product(*params_values))
    for value_set in params_values_product:
        kwargs = {name: value for name, value in zip(params_names, value_set)}
        print(kwargs)
        pipeline.set_params(**kwargs).fit(feature_train, label_train)
        test_classifier(pipeline, data_dict, features_list, folds)

