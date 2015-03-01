#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

from numpy import log

from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

"""
Features
"""

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
poi = ["poi"]

### Separate list to apply PCA to each one. Emails looks for underlying
### feature of constant communication between POI's
features_email = [
                "poi_ratio_messages",
                "from_messages",
                "from_poi_to_this_person",
                "from_this_person_to_poi",
                "shared_receipt_with_poi",
                "to_messages"]
                 
    
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

features_list = poi + features_email + features_financial

    
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
        
def add_features(data_dict, features_list, financial_log=False):
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
        except:
            data_dict[name]['poi_ratio_messages'] = 'NaN'
        
        # If feature is financial, add another variable with log transformation.
        if financial_log:
            for feat in features_financial:
                try:
                    data_dict[name][feat + '_log'] = log(data_dict[name][feat])
                except:
                    data_dict[name][feat + '_log'] = 'NaN'
        
    return data_dict


def get_pca_features(data, email_components=1, financial_components=2):
    """
    Function calculates PCA for email and financial features sep.
    """
    from sklearn.decomposition import RandomizedPCA
    
    # Separate the POI and newly calculates features
    initial_feature = data[:, [0,1]]

    ### PCA new features
    data_email = data[:, [1,5]]
    data_financial = data[:, [6,20]]

    pca_email = RandomizedPCA(n_components=email_components, whiten=True)
    pca_email.fit(data_email)
    features_email_pca = pca_email.transform(data_email)

    pca_finance = RandomizedPCA(n_components=financial_components, whiten=True)
    pca_finance.fit(data_financial)
    features_finance_pca = pca_finance.transform(data_financial)

    from numpy import c_
    pca_features = c_[features_finance_pca, features_email_pca]
    data = c_[initial_feature, pca_features]
    
    return data


def scale_features(features):
    """
    Split and scale features. Returns two Numpy nd arrays, labels and features.
    """
    
    # scale features via min-max
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(features)
    
    return features


# select K best features
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#
#number_of_features = 10
#features = SelectKBest(chi2, number_of_features).fit_transform(features, labels)


### machine learning goes here!
### please name your classifier clf for easy export below

def setup_clf_list():
    """
    Instantiates all classifiers of interstes to be used.
    """
    from sklearn.pipeline import Pipeline
    
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.lda import LDA
    from sklearn.neural_network import BernoulliRBM
    
    # List of tuples of a classifier and its parameters.
    clf_list = []

    #
    clf_naive = GaussianNB()
    params_naive = {}
    clf_list.append( (clf_naive, params_naive) )

    #
    clf_tree = DecisionTreeClassifier()
    params_tree = {}
    clf_list.append( (clf_tree, params_tree) )

    #
    clf_linearsvm = LinearSVC()
    params_linearsvm = {}
    clf_list.append( (clf_linearsvm, params_linearsvm) )

    #
    clf_adaboost = AdaBoostClassifier()
    params_adaboost = {}
    clf_list.append( (clf_adaboost, params_adaboost) )

    #
    clf_random_tree = RandomForestClassifier()
    params_random_tree = {}
    clf_list.append( (clf_random_tree, params_random_tree) )

    #
    clf_knn = KNeighborsClassifier()
    params_knn = {"n_neighbors":[2, 5], "p":[2,3]}
    clf_list.append( (clf_knn, params_knn) )

    #
    clf_log = LogisticRegression()
    params_log = {"C":[10**18], "tol": [10**-21]}
    clf_list.append( (clf_log, params_log) )

    #
    clf_lda = LDA()
    params_lda = {}
    clf_list.append( (clf_lda, params_lda) )
    
    #
    logistic = LogisticRegression()
    rbm = BernoulliRBM(n_components=2)
    clf_rbm = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    params_rbm = {}
    clf_list.append( (clf_rbm, params_rbm) )
    
    return clf_list



def optimize_clf(clf, params, features_train, labels_train):
    """
    Given a classifier and its parameters, uses GridSearchCV to
    find the optimal parameters. Returns 
    """
    
#    print "*"*40
#    print "Fitting the classifier to the training set"
#    print clf
    
#    t0 = time()
    clf = GridSearchCV(clf, params)
    clf = clf.fit(features_train, labels_train)
    
#    print "done in %0.3fs" % (time() - t0)
#    print "Best estimator found by grid search:"
#    print clf.best_estimator_
#    print "*"*40
    
    return clf.best_estimator_


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

def train_unsupervised_clf(features_train, labels_train):
    """
    """
    from sklearn.cluster import KMeans
    
    clf_kmeans = KMeans(n_clusters=2, tol=0.001)
    clf_kmeans.fit( features_train )
    
    return [clf_kmeans]

def train_clf(features_train, labels_train):
    """
    """
    
    clf_supervised = setup_clf_list()
    clf_supervised = optimize_clf_list(clf_supervised, features_train, labels_train)
    
    clf_unsupervised = train_unsupervised_clf(features_train, labels_train)
    
    return clf_supervised + clf_unsupervised
    


###############################################################################
# Quantitative evaluation of the model quality on the test set

def evaluate_clf(clf, features_test, labels_test):
    
#    print "\nPredicting the people names on the testing set"
#    print clf
#    t0 = time()
    labels_pred = clf.predict(features_test)
#    print "done in %0.3fs" % (time() - t0)

#    print classification_report(labels_test, labels_pred)
    
    f1 = f1_score(labels_test, labels_pred)
    recall = recall_score(labels_test, labels_pred)
    precision = precision_score(labels_test, labels_pred)
    return f1, recall, precision

def evaluate_clf_list(clf_list, features_test, labels_test):
    
    clf_with_scores = []
    for clf in clf_list:
        f1, recall, precision = evaluate_clf(clf, features_test, labels_test)
        clf_with_scores.append( (clf, f1, recall, precision) )
        
#    orederd_clf = sorted(clf_with_scores, key = lambda tup: tup[1], reverse=True)
    return clf_with_scores


def evaluation_loop(features, labels, num_iters=1000, test_size=0.3):
    """
    Run evaluation metrics multiple times, exactly iteration_count, so as to
    get a better idea of which classifier is doing better with different 
    data splits.
    """
    from numpy import asarray
    
    evaluation_matrix = [[] for n in range(10)]
    for i in range(num_iters):

        #### Split data into training and test sets
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size)

        ### Tain all models
        clf_list = train_clf(features_train, labels_train)

        for i, clf in enumerate(clf_list):
            scores = evaluate_clf(clf, features_test, labels_test)
            evaluation_matrix[i].append(scores)
    
    # Make a copy of the classifications list. Just want the structure.
    summary_list = {}
    for i, col in enumerate(evaluation_matrix):
        summary_list[clf_list[i]] = ( sum(asarray(col)) )
    
    ordered_list = sorted(summary_list.keys() , key = lambda k: summary_list[k][0], reverse=True)
    return ordered_list, summary_list
    
def main():
    
    # import build_email_features
    data_dict = load_preprocess_data()
    data_dict = add_features(data_dict, features_list)

    ### if you are creating any new features, you might want to do that here
    ### store to my_dataset for easy export below
    my_dataset = data_dict

    ### these two lines extract the features specified in features_list
    ### and extract them from data_dict, returning a numpy array
    data = featureFormat(my_dataset, features_list)
    
    data = get_pca_features(data, email_components=1, financial_components=2)
    
    ### split into labels and features (this line assumes that the first
    ### feature in the array is the label, which is why "poi" must always
    ### be first in features_list
    labels, features = targetFeatureSplit(data)
    features = scale_features(features)
    
    # Another way to run the code for 1 iteration.
#    #### Split data into training and test sets
#    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30)
#    
#    ### ML
#    clf_list = train_clf(features_train, labels_train)
#    
##    clf_evaluated = evaluate_clf_list(clf_list, features_test, labels_test)
##    print clf_evaluated
#    return clf_evaluated[0][0], data_dict


    ordered_list, summary_list = evaluation_loop(features, labels, num_iters=100, test_size=0.3)
#    print ordered_list
#    print summary_list

    return ordered_list, summary_list, data_dict

ordered_list, summary_list, data_dict = main()
clf = ordered_list[0]
f1_score = summary_list[clf][0]
print "Best classifier is ", clf
print "With F1 score of: ", f1_score


### dump your classifier, dataset and features_list so
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
