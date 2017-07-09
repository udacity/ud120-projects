
##1
```
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
```

1. Eliminate all features that does not really have any data

| Feature   |  Zero/Nan total Count | % of all data points is either NaN/0 |
| --------- | --------------------- | ------------------------------------ |
| loan_advances |   141  | 97.9166666667 |
| director_fees |   128  | 88.8888888889 |
| restricted_stock_deferred |   142  | 88.1944444444 |
| poi |   126  | 87.5 |
| deferral_payments |   107  | 73.6111111111 |
| deferred_income |   144  | 66.6666666667 |
| long_term_incentive |   79  | 54.8611111111 |
| from_this_person_to_poi |   78  | 54.1666666667 |
| from_poi_to_this_person |   70  | 48.6111111111 |
| bonus |   63  | 43.75 |
| shared_receipt_with_poi |   58  | 40.2777777778 |
| to_messages |   58  | 40.2777777778 |
| from_messages |   58  | 40.2777777778 |
| other |   52  | 36.1111111111 |
| salary  |   50  | 34.7222222222 |
| expenses  |   50  | 34.7222222222 |
| exercised_stock_options |   43  | 29.8611111111 |
| restricted_stock  |   36  | 25.0 |
| total_payments  |   20  | 13.8888888889 |
| total_stock_value |   20  | 13.8888888889 |

Chose to remove everything where more then 60% 0 or NaN data points, where a 0 does not make sense. Which is the case for POI.


2. Using SelectKBest To select the top features

![Chart](./SelecktKBest_features.png "Best features")

Conclusion: I will use SelectKBest as part of the pipline and will include all features where the data has mare than 60% 0/NaN values.

## 2.
```
### Task 2: Remove outliers
```

Plot all data points Salary/Bonus, removed the total data point as it is the sum of all the other features.

![outlier](./outlier_with_total.png "With outlier")
![no_outlier](./outlier_without_total.png "With outlier")

## 3.
```
### Task 3: Create new feature(s)
```

Added two new features giving the ratio of emails going to/from this person to a POI
"from_this_person_to_poi", "from_poi_to_this_person", "shared_receipt_with_poi", "to_messages", "from_messages"

| Feature   |  Zero/Nan total Count | % of all data points is either NaN/0 |
| --------- | --------------------- | ------------------------------------ |
| ratio_from_to_poi |   78  | 54.1666666667 |
| ratio_to_from_poi |   70  | 48.6111111111 |

Compared to the best existing features in feature selection

![Chart](./new_features_feature_selection.png "New features")

Conclusion is that the feature ```from_this_person_to_poi``` looks to perform better than ```ratio_to_from_poi```


## 4.
```
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
```
