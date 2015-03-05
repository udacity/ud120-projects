# Investigating Enrons scandal using Machine Learning
===================================================
By: Sebastian Ibarguen
@sebasibarguen

## Introduction
> [In addition to being the largest bankruptcy reorganization in American history at that time, Enron was cited as the biggest audit failure](http://en.wikipedia.org/wiki/Enron_scandal)

From a $90 price per share, to a $1 value represents the huge value loss and scam that happened in Enron. This case has been
a point of interest for machine learning analysis and trying to figure out what went wrong and how to avoid it in the future. It would be of great value to find a model that could potentially predict these types of events before much damage is done. Corporate governance, the stock market, and even the Government would be quite interested in a machine learning model that could signal potential fraud detections before hand.

## Enron Data
The interesting and hard part of the dataset is that it's very skewed, given that from the 146 there are only 11 people
or data points labeled as POI's or guilty of fraud. We are interested in labeling every person in the dataset
into either a POI or a non-POI. More than that, if we can assign a probability to a person to see what is the chance she is POI, is a more reasonable model given there is always uncertainty.


## Feature Processing
All features in the dataset are either financial data or features extracted from emails. Financial data includes features like salary and bonus while the email features include number of messages written to whom.

There are 2 clear outliers in the data, **TOTAL** and **THE TRAVEL AGENCY IN THE PARK**. The first one seems to be the sum total of all the other data points, while the second outlier is quite bizarre. Both these outliers are removed from the dataset for all the analysis. Also all features are scaled using the **MinMaxScaler**.

### New features
From the initial dataset, 5 new features where added, you can find more details in the table below:

|Feature | Description     |
|--------|-----------------|
|Ratio of POI messages | POI related messages divided over the total messages from the person |
|Log of financials (multiple) | Financial variables with logarithmic transformation |

The reasoning behind the **ratio of POI messages** is that we expect that POI's contact each other relatively more often than with non-POI's and the relationship might be non-linear. We also can expect that the financial gains for POI's is actually non-linear, that is why applying a logarithmic transformation should improve many algorithms.

### PCA
It's quite reasonable to think that all the **email** features we have, 5 initial features plus 1 computed feature, really represent 1 underlying feature or principal component like higher communication between POI's. The same goes for the financial features, which would really be to measure the POI's corruption via big money gains. In other words, we expect that a POI has a higher money gain compared to a non-POI, and that all the financial features are really trying to measure this underlying one. By tuning the parameters, we get the best classification results with 2 email components and 3 financial components. This means that from the initial 6 email features, we are reducing them to 2.

It is very interesting to see that applying PCA to the features helps out some classifiers more than others, and actually changes the top classifier. Without PCA the best estimator is Logistic Regression, with PCA, it's Linear Discriminant Analysis (LDA).


## Algorithms selection and tuning
For the analysis of the data, a total of 10 classifiers where tried out, which include:
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

To tune the overall performance, there were automatic parameter tunings, and manual ones as well. The automatically tuned parameters where around each model using the **GridSearchCV** took from SkLearn. The manual tuning occurred in the following ways:
1. Including the PCA features
2. Adding/removing features
3. Scaling features

For the most part, PCA did not improve the performance of the models in general. Adding and removing features did help out,
specially adding the ratio of messages related to POI and all messages, the new financial variables transformed and removing many features. Scaling features does help out, and even helps determine the winning algorithm. When the features are scaled, normally GNB is the best algorithm, when the features are not scaled, Linear SVC takes the lead, but overall GNB performs better.

### Optimization
All the supervised learning algorithms (9/10) where optimized using **GridSearchCV**. The general
process was:
> build list of classifier with parameters > optimize each classifier with training data > evaluate all the classifiers > compare f1, recall and precision scores > choose the best classifier


### Validation and Performance
To validate the performance of each algorithm, recall, precision and F1 scores where calculated for each one. You can find below a summary of the scores of the top algorithms.

|Feature | F1 Score | Recall | Precision |
|--------|----------|--------|-----------|
|Gaussian Naive Bayes         | 33.83 | 91.40 | 21.21 |
|Linear Discriminant Analysis | 30.50 | 23.41 | 56.07 |
|Decision Tree                | 27.19 | 29.25 | 28.96 |
|AdaBoost                     | 22.75 | 22.08 | 28.09 |
*Results WILL vary. There is some randomness in the data splitting*


The 3 best classifiers were Gaussian Naive Bayes (GNB), Linear Discrimination Analysis (LDA) and Logistic Regression.
Given that the code process is automated and some inherit randomness is present (coming from the way the data is split), either the LDA or GNB could come out as the best estimators. The most frequent algorithm was GNB with an F1 score of around 30. The Decision tree and Adaboost algorithms also performed relatively well at the end. If your interest and priority was to identify POI's then probably the best algorithm for the task would be LDA, given that it has a significantly higher precision.

In the end, a simple Gaussian Naive Bayes model proved to be up to the task. It was the best model overall, when evaluated on the F1 score.

## Discussion and Conclusions
This was just a starting point analysis for classifying Enron employees. The results should not be taken too seriously and more advanced models should be used. Possibilities for future research could be to include more complex pipelines for the data, or even Neural Networks. Here we tried a basic neural network, but the SkLearn library is very limited in what it has to offer in this regard.
