# Change Log
	* All notable changes to this project will be documented in this file.
	* Lecture wise File Specific changes are also mentioned.


## Common Changes to All Files
- Migrated all codes to Python-3.
- Updated Shabang to '#!/usr/bin/python3'.
- Updated Libraries to support Python 3.6 or higher.
- Added Python-3 version of Course Quiz questions. (In Some Cases).
- Updated 'readme.md'
- Updated 'requirements.txt'.
- Updated '.gitignore'.
- Added 'Changelog.md'

## Lecture 1 : Intro to Machine Learning
### tools/startup.py
- Updated printing the file size from '423 MB' to '1.82 GB' as the Dataset has been Updated.
- Updated code for downloading 'enron_mail_20150507.tar.gz' using 'requests' instead of 'urllib'.
- Updated code for extracting 'enron_mail_20150507.tar.gz' for Python-3.

### tools/email_preprocess.py
- 'joblib' used instead of 'Pickle' and 'cPickle'.
- 'model_selection.train_test_split' used instead of 'cross_validation.train_test_split'.


## Lecture 2 : Naive Bayes
### naive_bayes/nb_author_id.py
- Added Python-3 code to print Training and Predicting Time for Course Quiz.


## Lecture 3 : SVM
### svm/svm_author_id.py
- Added the Fix for Indexing Issue in slicing 1% of Training Data.


## Lecture 4 and Lecture 5
- No Special Changes


## Lecture 6 : Dataset and Questions
### datasets_questions/explore_enron_data.py
- 'joblib' used instead of 'Pickle'.

### tools/feature_format.py
- 'joblib' used instead of 'Pickle'


## Lecture 7 : Regression
### regression/finance_regression.py
- 'joblib' used instead of 'Pickle'.
- Updated reading mode of '.pkl' file from 'r' to 'rb' to resolve file reading issue.
- 'model_selection.train_test_split' used instead of 'cross_validation.train_test_split'.
- Added 'sort_keys' parameter to Line 26 as mentioned in the Course for Python-3 Compatibility.


## Lesson 8 : Outliers
### outliers/enron_outliers.py
- 'joblib' used instead of 'Pickle'.
- Updated reading mode of '.pkl' file from 'r' to 'rb' to resolve file reading issue.


### outliers/outlier_removal_regression.py
- 'joblib' used instead of 'Pickle'.
- Updated reading mode of '.pkl' file from 'r' to 'rb' to resolve file reading issue.
- 'model_selection.train_test_split' used instead of 'cross_validation.train_test_split'.


## Lesson 9 : Clustering
### k_mean/k_means_cluster.py
- 'joblib' used instead of 'Pickle'.
- Updated reading mode of '.pkl' file from 'r' to 'rb' to resolve file reading issue.


## Lecture 10
- No Special Changes


## Lecture 11 : Text Learning
### tools/parse_out_email_text.py
- Updated 'string' syntax for Python-3 Compatibility.

### text_learning/vectorize_text.py
- 'joblib' used instead of 'Pickle'.


## Lecture 12 : Feature Selection
### feature_selection/find_signature.py
- 'joblib' used instead of 'Pickle'.
- 'model_selection.train_test_split' used instead of 'cross_validation.train_test_split'.


## Lecture 13 : PCA
### pca/eigenfaces.py
- 'model_selection.train_test_split' used instead of 'cross_validation.train_test_split'.


## Lecture 14 : Validation
- No Special Changes


## Lecture 15 : Evaluation Metrics
- No Special Changes


## Lecture 16 : Trying it all Together
- No Special Changes


## Lecture 17 : Final Project
- No Special Changes