import pickle
import sys
import matplotlib.pyplot
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# {
# 'salary': 243293,
# 'to_messages': 1045,
# 'deferral_payments': 'NaN',
# 'total_payments': 288682,
# 'exercised_stock_options': 5538001, !!! KBEST
# 'bonus': 1500000,
# 'restricted_stock': 853064,
# 'shared_receipt_with_poi': 1035,?? Weak
# 'restricted_stock_deferred': 'NaN',
# 'total_stock_value': 6391065, !!!! KBEST
# 'expenses': 34039, ?? Good higher than usual
# 'loan_advances': 'NaN',
# 'from_messages': 32,
# 'other': 11350,
# 'from_this_person_to_poi': 21,
# 'poi': True,
# 'director_fees': 'NaN',
# 'deferred_income': -3117011,
# 'long_term_incentive': 1617011,
# 'email_address': 'kevin.hannon@enron.com',
# 'from_poi_to_this_person': 32}

# features_list = ['long_term_incentive', 'bonus', 'poi'] # You will need to use more features
features_list = ["poi", "salary", "to_messages", "deferral_payments", "total_payments", "exercised_stock_options", "bonus", "restricted_stock", "shared_receipt_with_poi", "restricted_stock_deferred", "total_stock_value", "expenses", "loan_advances", "from_messages", "other", "from_this_person_to_poi", "director_fees", "deferred_income", "long_term_incentive", "from_poi_to_this_person"]

data_dict.pop("TOTAL", 0)
data = featureFormat(data_dict, features_list)
### your code below

for point in data:
    p = point[5]
    bonus = point[6]
    color = 'r' if point[0] == 1 else 'b'
    matplotlib.pyplot.scatter( p, bonus, c = color)

matplotlib.pyplot.xlabel("exercised_stock_options")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# Perform feature selection

point_pois = []
data_featuers = []
for point in data:
    data_featuers.append(np.array(point))
    point_pois.append(point[0])

print data_featuers
selector = SelectKBest(f_classif, k=10)
selector.fit(data_featuers, np.array(point_pois))

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
matplotlib.pyplot.bar(range(len(features_list)), scores)
matplotlib.pyplot.xticks(range(len(features_list)), features_list, rotation='vertical')
matplotlib.pyplot.show()
