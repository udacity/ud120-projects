#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

from numpy import log

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
    
    ### load the dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    
    ### reoving outliers
    outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
    # 'LOCKHART EUGENE E' all values are NaN
    for outlier in outliers:
        data_dict.pop(outlier, 0)
    
    return data_dict
        
def add_features(data_dict, features_list, financial_log=False):
    
    # Ratio
    for name in data_dict:
        
        try:
            total_messages = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
            poi_related_messages = data_dict[name]["from_poi_to_this_person"] +\
                                    data_dict[name]["from_this_person_to_poi"] +\
                                    data_dict[name]["shared_receipt_with_poi"]
            poi_ratio = 1.* poi_related_messages / total_messages
            data_dict[name]['poi_ratio_messages'] = poi_ratio
        except:
            data_dict[name]['poi_ratio_messages'] = 'NaN'
        
        if financial_log:
            for feat in features_financial:
                try:
                    data_dict[name][feat + '_log'] = log(data_dict[name][feat])
                except:
                    data_dict[name][feat + '_log'] = 'NaN'
        
    return data_dict
    


#def extract_text_features():
#    import build_email_features
data_dict = load_preprocess_data()
data_dict = add_features(data_dict, features_list)

### if you are creating any new features, you might want to do that here
### store to my_dataset for easy export below
my_dataset = data_dict


### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)
initial_feature = data[:, [0,1]]
#print initial_feature

### PCA new features
from sklearn.decomposition import RandomizedPCA
from numpy import log

data_email = data[:, [1,5]]
data_financial = data[:, [6,20]]

print data_email.shape, data_financial.shape

pca_email = RandomizedPCA(n_components=1, whiten=True).fit(data_email)
features_email_pca = pca_email.transform(data_email)

pca_finance = RandomizedPCA(n_components=2, whiten=True).fit(data_financial)
features_finance_pca = pca_finance.transform(data_financial)

from numpy import c_

pca_features = c_[features_finance_pca, features_email_pca]

data = c_[initial_feature, pca_features]

### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)




# scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


# select K best features
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#
#number_of_features = 10
#features = SelectKBest(chi2, number_of_features).fit_transform(features, labels)

#### Split data into training and test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30)


"""
Algorithm selection
"""


### machine learning goes here!
### please name your classifier clf for easy export below
from time import time
from sklearn.grid_search import GridSearchCV

#
from sklearn.naive_bayes import GaussianNB

#
from sklearn.tree import DecisionTreeClassifier

#
from sklearn.svm import LinearSVC 
clf_linearsvm = LinearSVC()

#
from sklearn.ensemble import AdaBoostClassifier

#
from sklearn.ensemble import RandomForestClassifier

#
from sklearn.neighbors import KNeighborsClassifier
clf_knc = KNeighborsClassifier()

#
from sklearn.cluster import KMeans
clf_kmeans = KMeans(n_clusters=2)

#
from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression(C=10**18, tol=10**-21)

from sklearn.lda import LDA
clf_lda = LDA()

# print "Fitting the classifier to the training set"
# t0 = time()
# parameters = {"kernel":("linear", "rbf"), "C":[1, 10]}
# params_tree = {}
# params_knn = {"n_neighbors":[2, 5], "p":[2,3]}
#
# clf = GridSearchCV( KNeighborsClassifier(), params_knn)
# clf = clf.fit(features_train, labels_train)
# print "done in %0.3fs" % (time() - t0)
# print "Best estimator found by grid search:"
# print clf.best_estimator_


clf_list = [clf_log, 
            clf_kmeans,
            clf_linearsvm
           ]

clf = clf_kmeans
clf.fit(features_train, labels_train)
###############################################################################
# Quantitative evaluation of the model quality on the test set

from sklearn.metrics import classification_report

print "Predicting the people names on the testing set"
t0 = time()
labels_pred = clf.predict(features_test)
print "done in %0.3fs" % (time() - t0)

#print labels_test, "\n", labels_pred
print classification_report(labels_test, labels_pred)


### dump your classifier, dataset and features_list so
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
