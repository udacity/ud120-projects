#!/usr/bin/python

import sys
import pickle
import matplotlib
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary'] # You will need to use more features

### Load the dictionary containing the dataset
# with open("final_project_dataset.pkl", "r") as data_file:
#     data_dict = pickle.load(data_file)

# data_dict = pickle.load( open( "final_project_dataset.pkl", "rb" ) )

with open('final_project_dataset.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)



### Task 2: Remove outliers
identified_outliers = ["TOTAL", "LAVORATO JOHN J", "MARTIN AMANDA K", "URQUHART JOHN A", "MCCLELLAN GEORGE", "SHANKMAN JEFFREY A", "WHITE JR THOMAS E", "PAI LOU L", "HIRKO JOSEPH"]
for outlier in identified_outliers:
    data_dict.pop(outlier)

### Task 3: Create new feature(s)
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']

# count data_points
data_points = len(data_dict)

# initialise counts
poi_count = 0
non_poi_count = 0


# print
non_poi_count = 0
print( "Data points:\t", data_points)
print( "Number of non POIs:\t", non_poi_count)
print( "Number of POIs:\t\t", poi_count)

print( "POI ratio:\t\t", poi_count/data_points)
print( "Total features:\t", len(data_dict[data_dict.keys()[0]]))
print( "Financial features:\t", len(financial_features))
print( "Email features:\t", len(email_features))
print( "")





def outlier_visualization(data):
    for point in data:
        f1 = point[0]
        f2 = point[1]
        matplotlib.pyplot.scatter(f1, f2 )
    
    matplotlib.pyplot.xlabel("Feature 1")
    matplotlib.pyplot.ylabel("Feature 2")
    matplotlib.pyplot.show()



def visualize_outliers():
    start = 1
    for i in range(2, len(financial_features)):
        outlier_visualization(financial_outliers, 1, i, 'salary', financial_features[i], start)
        start += 1
    start = 10

    for i in range(2, len(email_features)):
        outlier_visualization(email_outliers, 1, i, 'to_messages', email_features[i], start)
        start += 1


# outlier name
def get_outlier(feature, value):
    for person, features in data_dict.iteritems():
        if features[feature] == value:
            print("Outlier is:", person, features['poi'])





### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

financial_outliers = featureFormat(data_dict, financial_features)
email_outliers = featureFormat(data_dict, email_features)


#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# import
from sklearn.naive_bayes import GaussianNB

# create classifier
clf = GaussianNB()

#fit/train
clf.fit(features_train, labels_train)

# predict
pred = clf.predict(features_test)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection  import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

visualize_outliers()

dump_classifier_and_data(clf, my_dataset, features_list)



