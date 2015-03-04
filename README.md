# Investigating Enrons scandal using Machine Learning
===================================================

## Introduction
> [In addition to being the largest bankruptcy reorganization in American history at that time, Enron was cited as the biggest audit failure](http://en.wikipedia.org/wiki/Enron_scandal)

From a $90 price per share, to a $1 value represents the huge value loss and scam that happened in Enron. This has been
a point of interest for analyzing and trying to figure out what went wrong and how to avoid it in the future. In particular
Machine Learning as a field has analyzed all the publicly available data.

## Enron Data
The interesting and hard part of the dataset is that it's very skewed, given that from the 146 there are only 11 people
or data points labeld as POI's or guilty of fraud. We are interesteed in labeling every person in the dataset
into either a POI or a non-POI. More than that, assigning a probability if a person is POI, is a more reasonable model given
there is always uncertainty given the features we have.


## Feature Processing
All features in the dataset are either financial data or features extracted from emails.

### Outliers
There are 2 clear outliers in the data, **TOTAL** and **THE TRAVEL AGENCY IN THE PARK**. The first one seems to be the sum total of all the other data points, while the second outlier is quite bizarre. Both these outliers are removed from the dataset for all the analysis.

### New features
From the initial dataset, 5 new features where added, you can find more details in the table below:

|Feature | Description     |
|--------|-----------------|
|Ratio of POI messages | POI related messages devided over the total messages from the person |
|Log of financials (multiple) | Financial variables with logarithmic transformation |

The reasoning behind the **ratio of POI messages** is that we expect that POI's contact each other relatively more often than with non-POI's and the relationship might be non-linear. We also can expect that the financial gains for POI's is actually non-linear, that is
why applying a logarithmic transformation should improve many algorithms.

### PCA
It's quite reasonable to hypothesis that all the email features we have, 5 initial features plus 1 computed feature, really represent 1 underlying feature or principal component like higher communication between POI's. The same goes for the financial features, which would really be to measure the POI's corruption via big money gains. In other words, we expect that a POI has a higher money gain compared to a non-POI, and that all the financial features are really trying to measure this underlying one. By tuning the parameters, we get the best classification results with 2 email components and 3 financial components. This means that from the initial 6 email features, we are reducing them to 2.

Interesting to see that applying PCA helps out some classifiers more and actually changes the top classifier. Without PCA the best estimator is Logistsic Regression, with PCA, it's Linear Discriminant Analysis (LDA).

### Scaling
All features are scaled using the **MinMaxScaler**. This is mostly to help out certain algorithms like KMeans.


## Algorithms selection and tuning
For the analysis of the data, a total of 10 classifiers where tried out, including:
- Logistic Regression
- Linear Discriminant Analysis
- Decision Tree Classifier
- Gaussian Naive Bayes
- Linear Support Vector Classifier (LinearSVC)
- AdaBoost
- Random Forrest Tree Classifier
- K Nearest Neighbor
- KMeans
- Bernoulli RBM (together with Logistic Regression)

The object of the algorithm is to classify and find out which people are more likely to be POI's. There are clearly
2 categories we are looking to label the data.

### Optimization
All the supervised learning algorithms (9/10) where optimized using **GridSearchCV**. The general
process was:
> build list of classifier with parameters > optimize each classifier with training data > evaluate all the classifiers > compare f1, recall and precision scores > choose the best classifier



### Validation and Performance
To validate the performance of each algorithm, recall, precision and F1 scores where used. Find below a summary of the scores.

|Feature | F1 Score | Recall | Precision |
|--------|----------|--------|-----------|
|Logistic Regression |
|Linear Discriminant Analysis |30.50|23.41|56.07|
|Decision Tree Classifier |
|Gaussian Naive Bayes |
|Linear Support Vector Classifier |15.83910534|10.26785714|44.5|
|AdaBoost |
|Randome Forrest Tree Classifier |
|K Nearest Neighbor |
|KMeans |
|Bernoulli RBM |


The 3 best classifiers were Linear Discrimination Analysis (LDA), Gaussian Naive Bayes (GNB) and Logistic Regression.
Given that the pipeline including optimization and some randmoness, when running the full code, either LDA or GNB
would come out as the best estimators.  

## Discussion and Conclusions
