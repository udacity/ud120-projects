#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from numpy import log
from numpy import sqrt
from numpy import float64
from numpy import nan

from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.neural_network import BernoulliRBM

from sklearn.cluster import KMeans
"""
Features
"""

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
poi = ["poi"]

### Separate list to apply PCA to each one. Emails looks for underlying
### feature of constant communication between POI's
features_email = [
                "from_messages",
                "from_poi_to_this_person",
                "from_this_person_to_poi",
                "shared_receipt_with_poi",
                "to_messages"
                ]


### Financial features might have underlying features of bribe money
features_financial = [
                "bonus",
                "deferral_payments",
                "deferred_income",
                "director_fees",
                "exercised_stock_options",
                "expenses",
                "loan_advances",
                "long_term_incentive",
                "other",
                "restricted_stock",
                "restricted_stock_deferred",
                "salary",
                "total_payments",
                "total_stock_value"
                ]

features_new = [

                # Email
                "poi_ratio_messages",
                # "poi_ratio_messages_square",

                # Log Feats
                "total_payments_log",
                "salary_log",
                "bonus_log",
                "total_stock_value_log",
                "exercised_stock_options_log",

                # squared feats
                "total_payments_squared",
                "total_stock_value_squared",
                "exercised_stock_options_squared",
                "salary_squared",


]


features_list = poi + features_email + features_financial + features_new

NAN_value = 'NaN'

def load_preprocess_data():
    """
    Loads and removes outliers from the dataset. Returns the data as a
    Python dictionary.
    """

    ### load the dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

    ### reoving outliers
    outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
    # 'LOCKHART EUGENE E' all values are NaN
    for outlier in outliers:
        data_dict.pop(outlier, 0)

    return data_dict

def add_features(data_dict, features_list, financial_log=False, financial_squared=False):
    """
    Given the data dictionary of people with features, adds some features to

    """
    for name in data_dict:

        # Add ratio of POI messages to total.
        try:
            total_messages = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
            poi_related_messages = data_dict[name]["from_poi_to_this_person"] +\
                                    data_dict[name]["from_this_person_to_poi"] +\
                                    data_dict[name]["shared_receipt_with_poi"]
            poi_ratio = 1.* poi_related_messages / total_messages
            data_dict[name]['poi_ratio_messages'] = poi_ratio
            data_dict[name]['poi_ratio_messages_squared'] = poi_ratio ** 2
        except:
            data_dict[name]['poi_ratio_messages'] = NAN_value

        # If feature is financial, add another variable with log transformation.
        if financial_log:
            for feat in features_financial:
                try:
                    data_dict[name][feat + '_log'] = Math.log(data_dict[name][feat] + 1)
                except:
                    data_dict[name][feat + '_log'] = NAN_value

        # Add squared features
        if financial_squared:
            for feat in features_financial:
                try:
                    data_dict[name][feat + '_squared'] = Math.square(data_dict[name][feat]+1)
                except:
                    data_dict[name][feat + '_squared'] = NAN_value

    # print "finished"
    return data_dict


def transform_pca_pipeline(clf_list):
    """
    Function takes a classifier list and returns a list of piplines of the
    same classifiers and PCA.
    """

    pca = PCA()
    params_pca = {"pca__n_components":[2, 3, 4, 5, 10, 15, 20], "pca__whiten": [False]}

    for i in range(len(clf_list)):

        name = "clf_" + str(i)
        clf, params = clf_list[i]

        # For GridSearch to work with pipeline, the params have to have
        # double underscores between specific classifier and its parameter.
        new_params = {}
        for key, value in params.iteritems():
            new_params[name + "__" + key] = value

        new_params.update(params_pca)
        clf_list[i] = (Pipeline([("pca", pca), (name, clf)]), new_params)

    return clf_list


def scale_features(features):
    """
    Scale features using the Min-Max algorithm
    """

    # scale features via min-max
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(features)

    return features


def setup_clf_list():
    """
    Instantiates all classifiers of interstes to be used.
    """
    # List of tuples of a classifier and its parameters.
    clf_list = []

    #
    # clf_naive = GaussianNB()
    # params_naive = {}
    # clf_list.append( (clf_naive, params_naive) )

    #
    clf_tree = DecisionTreeClassifier()
    params_tree = { "min_samples_split":[2, 5, 10, 20],
                    "criterion": ('gini', 'entropy')
                    }
    clf_list.append( (clf_tree, params_tree) )

    #
    clf_linearsvm = LinearSVC()
    params_linearsvm = {"C": [0.5, 1, 5, 10, 100, 10**10],
                        "tol":[10**-1, 10**-10],
                        "class_weight":['auto']

                        }
    clf_list.append( (clf_linearsvm, params_linearsvm) )

    #
    clf_adaboost = AdaBoostClassifier()
    params_adaboost = { "n_estimators":[20, 25, 30, 40, 50, 100]
                        }
    clf_list.append( (clf_adaboost, params_adaboost) )

    #
    clf_random_tree = RandomForestClassifier()
    params_random_tree = {  "n_estimators":[2, 3, 5],
                            "criterion": ('gini', 'entropy')
                            }
    clf_list.append( (clf_random_tree, params_random_tree) )

    #
    clf_knn = KNeighborsClassifier()
    params_knn = {"n_neighbors":[2, 5], "p":[2,3]}
    clf_list.append( (clf_knn, params_knn) )

    #
    clf_log = LogisticRegression()
    params_log = {  "C":[0.05, 0.5, 1, 10, 10**2,10**5,10**10, 10**20],
                    "tol":[10**-1, 10**-5, 10**-10],
                    "class_weight":['auto']
                    }
    clf_list.append( (clf_log, params_log) )

    #
    clf_lda = LDA()
    params_lda = {"n_components":[0, 1, 2, 5, 10]}
    clf_list.append( (clf_lda, params_lda) )

    #
    logistic = LogisticRegression()
    rbm = BernoulliRBM()
    clf_rbm = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    params_rbm = {
        "logistic__tol":[10**-10, 10**-20],
        "logistic__C":[0.05, 0.5, 1, 10, 10**2,10**5,10**10, 10**20],
        "logistic__class_weight":['auto'],
        "rbm__n_components":[2,3,4]
    }
    clf_list.append( (clf_rbm, params_rbm) )

    return clf_list



def optimize_clf(clf, params, features_train, labels_train, optimize=True):
    """
    Given a classifier and its parameters, uses GridSearchCV to
    find the optimal parameters. Returns
    """

    if optimize:
        scorer = make_scorer(f1_score)
        clf = GridSearchCV(clf, params, scoring=scorer)
        clf = clf.fit(features_train, labels_train)
        clf = clf.best_estimator_
    else:
        clf = clf.fit(features_train, labels_train)

    return clf


def optimize_clf_list(clf_list, features_train, labels_train):
    """
    Takes a list of tuples for classifiers and parameters, and returns
    a list of the best estimator optimized to it's given parameters.
    """

    best_estimators = []
    for clf, params in clf_list:
        clf_optimized = optimize_clf(clf, params, features_train, labels_train)
        best_estimators.append( clf_optimized )

    return best_estimators

def train_unsupervised_clf(features_train, labels_train, pca_pipeline):
    """
    Train unsupervised classifiers. Just KMeans for now.
    """

    clf_kmeans = KMeans(n_clusters=2, tol = 0.001)

    if pca_pipeline:
        pca = PCA(n_components=2, whiten=False)

        clf_kmeans = Pipeline([("pca", pca), ("kmeans", clf_kmeans)])

    clf_kmeans.fit( features_train )

    return [clf_kmeans]

def train_clf(features_train, labels_train, pca_pipeline=False):
    """
    """

    clf_supervised = setup_clf_list()

    if pca_pipeline:
        clf_supervised = transform_pca_pipeline(clf_supervised)

    clf_supervised = optimize_clf_list(clf_supervised, features_train, labels_train)
    clf_unsupervised = train_unsupervised_clf(features_train, labels_train, pca_pipeline)

    return clf_supervised + clf_unsupervised



###############################################################################
# Quantitative evaluation of the model quality on the test set

def evaluate_clf(clf, features_test, labels_test):


    labels_pred = clf.predict(features_test)

    f1 = f1_score(labels_test, labels_pred)
    recall = recall_score(labels_test, labels_pred)
    precision = precision_score(labels_test, labels_pred)
    return f1, recall, precision

def evaluate_clf_list(clf_list, features_test, labels_test):

    clf_with_scores = []
    for clf in clf_list:
        f1, recall, precision = evaluate_clf(clf, features_test, labels_test)
        clf_with_scores.append( (clf, f1, recall, precision) )

    return clf_with_scores


def evaluation_loop(features, labels, pca_pipeline=False, num_iters=1000, test_size=0.3):
    """
    Run evaluation metrics multiple times, exactly iteration_count, so as to
    get a better idea of which classifier is doing better with different
    data splits.
    """
    from numpy import asarray

    evaluation_matrix = [[] for n in range(9)]
    for i in range(num_iters):

        #### Split data into training and test sets
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size)


        ### Tain all models
        clf_list = train_clf(features_train, labels_train, pca_pipeline)

        for i, clf in enumerate(clf_list):
            scores = evaluate_clf(clf, features_test, labels_test)
            evaluation_matrix[i].append(scores)

    # Make a copy of the classifications list. Just want the structure.
    summary_list = {}
    for i, col in enumerate(evaluation_matrix):
        summary_list[clf_list[i]] = ( sum(asarray(col)) )

    ordered_list = sorted(summary_list.keys() , key = lambda k: summary_list[k][0], reverse=True)
    return ordered_list, summary_list


# import build_email_features
data_dict = load_preprocess_data()

### To test financial_log and financial_squared features, first turn them to True, then uncomment them up in features_new
data_dict = add_features(data_dict, features_list, financial_log=True, financial_squared=True)

### store to my_dataset for easy export below
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)
# features = scale_features(features)



### Select K best. Makes no sense to use when select k best
### when we are using PCA. (in some cases it might, but here it does not)
# k = 20
# k_best = SelectKBest(k=k)
# k_best.fit(features, labels)
#
# scores = k_best.scores_
# unsorted_pairs = zip(features_list[1:], scores)
# sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
# k_best_features = dict(sorted_pairs[:k])
# print "{0} best features: {1}\n".format(k, k_best_features.keys())
# print ""
# print k_best_features
#
# features_list = poi + k_best_features.keys()


####### UNCOMMENT THIS LINE TO RUN FULL CODE ####### Takes a long time to run if pca_pipeline is True.
# ordered_list, summary_list = evaluation_loop(features, labels, pca_pipeline = True, num_iters=10, test_size=0.3)

# print ordered_list
# print "*"*100
# print summary_list
# print "*"*100
#
# clf = ordered_list[0]
# scores = summary_list[clf]
# print "Best classifier is ", clf
# print "With scores of f1, recall, precision: ", scores


# Manually pick classifiers
clf_logistic = LogisticRegression(  C=10**20,
                                    penalty='l2',
                                    random_state=42,
                                    tol=10**-10,
                                    class_weight='auto')
clf_lda = LDA(n_components=2)

pca = PCA(n_components=20)

clf = Pipeline(steps=[("pca", pca), ("logistic", clf_logistic)])


test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
