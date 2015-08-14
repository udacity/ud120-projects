# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:14:05 2015

@author: jayantsahewal
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier


def ET_classif(features_df=None, labels_df=None):
  '''Scoring function to be used in SelectKBest feature selection class 
      object.
      
  This scoring function assigns varaible importances to the features
      passed in to it using the ExtraTreesClassifier. It then returns
      the features as two identical arrays mimicking the scores and 
      p-values arrays required by SelectKBest to pick the top K 
      features.
      
  Args:
      features_df: Pandas dataframe of features to be used to predict 
          using the ExtraTreesClassifier.
      labels_df: Pandas dataframe of the labels being predicted.
  Returns:
      Two identical arrays containing the feature importance scores
          returned for each feature by the ExtraTreesClassifier.
  '''
  reducer = ExtraTreesClassifier(n_estimators=500, bootstrap=False,
                                 oob_score=False, max_features=.10,
                                 min_samples_split=10, min_samples_leaf=2,
                                 criterion='gini', random_state=42)

  reducer.fit(features_df, labels_df)
  return reducer.feature_importances_, reducer.feature_importances_

def get_LogReg_pipeline():
  '''Make a pipeline for cross-validated grid search for the
      Logistic Regreesion Model.
  
  This function makes a pipeline which:
      1. Scales the features between 0-1
      2. Selects the KBest features using Anova F-value scoring for
          classification.
      3. Uses KBest features to reduce dimensionality further using PCA
      4. Using the resulting PCA components in Logistic Regression.
  
  '''
  pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                             ('selection', SelectKBest(score_func=f_classif)),
                             ('reducer', PCA()),
                             ('classifier', LogisticRegression())
                             ])

  return pipeline                        

def get_LogReg_params(full_search_params=False):
  '''Make a parameters dictionary for cross-validated grid search for the
      Logistic Regression Model.
  
  This function makes a parameter dictionary to search over.
  
  Parameters searched over include:
      SelectKBest:        
          1. k: Number of KBest features to select.
      PCA:
          1. n: Number of PCA components to retain.
          2. whiten: Boolean value whether to whiten the features during PCA.
      LogisticRegression:
          1. C: Value of the regularization constraint.
          2. class_weight: Over-/undersamples the samples of each class.
          3. tol: Tolerance for stopping criteria

  Args:
      full_search_params: Boolean value whether to search over an exhaustive 
          grid of params. (Can take a LONG time.)

  Returns:
      A dictionary of parameters to pass into an sk-learn grid-search 
          pipeline. Default parameters include only the final parameters 
          found through exhaustive searching.
  '''
  
  params = {'reducer__n_components': [.5], 
            'reducer__whiten': [False],
            'classifier__class_weight': ['auto'],
            'classifier__tol': [1e-64], 
            'classifier__C': [1e-3],
            'selection__k': [17]
            }
            
  if full_search_params:
    params = {'selection__k': [7, 9, 12, 17, 'all'],
              'classifier__C': [1e-07, 1e-05, 1e-2, 1e-1, 1, 10],
              'classifier__class_weight': [{True: 12, False: 1},
                                           {True: 10, False: 1},
                                           {True: 8, False: 1}],
              'classifier__tol': [1e-1, 1e-4, 1e-16,
                                  1e-64, 1e-256],
              'reducer__n_components': [1, 2, 3],
              'reducer__whiten': [True, False]
              }
               
  return params


def get_LSVC_pipeline():
  '''Make a pipeline for cross-validated grid search for the
      Linear Support Vector Machines Classifier.
  
  This function makes a pipeline which:
      1. Scales the features between 0-1
      2. Selects the KBest features using Anova F-value scoring for
          classification.
      3. Uses KBest features to reduce dimensionality further using PCA
      4. Using the resulting PCA components in Linear Support Vector
          Machines Classifier.
  
  '''
  pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                             ('selection', SelectKBest(score_func=f_classif)),
                             ('reducer', PCA()),
                             ('classifier', LinearSVC())
                             ])
                             
  return pipeline                        

def get_LSVC_params(full_search_params=False):
  '''Make a parameters dictionary for cross-validated grid search for the
      Linear Support Vector Machines Classifier.
  
  This function makes a parameter dictionary to search over.
  
  Parameters searched over include:
      SelectKBest:        
          1. k: Number of KBest features to select.
      PCA:
          1. n: Number of PCA components to retain.
          2. whiten: Boolean value whether to whiten the features during PCA.
      LogisticRegression:
          1. C: Value of the regularization constraint.
          2. class_weight: Over-/undersamples the samples of each class.
          3. tol: Tolerance for stopping criteria

  Args:
      full_search_params: Boolean value whether to search over an exhaustive 
          grid of params. (Can take a LONG time.)

  Returns:
      A dictionary of parameters to pass into an sk-learn grid-search 
          pipeline. Default parameters include only the final parameters 
          found through exhaustive searching.
  '''
  
  params = {'reducer__n_components': [.5], 
            'reducer__whiten': [True],
            'classifier__class_weight': ['auto'],
            'classifier__tol': [1e-32], 
            'classifier__C': [1e-5],
            'selection__k': [17]
            }
            
  if full_search_params:
      params = {'selection__k': [9, 12, 15, 17, 20],
                'classifier__C': [1e-15, 1e-5, 1e-2, 1e-1, 1, 10, 100],
                'classifier__class_weight': [{True: 12, False: 1},
                                             {True: 10, False: 1},
                                             {True: 8, False: 1},
                                             {True: 15, False: 1},
                                             {True: 20, False: 1}],
                'classifier__tol': [1e-1, 1e-2, 1e-4, 1e-8, 1e-16,
                                    1e-32, 1e-64, 1e-128, 1e-256],
                'reducer__n_components': [.25, .5, .75, .9, 1, 2, 3, 4, 5],
                'reducer__whiten': [True, False]
                }
               
  return params

  
def get_SVC_pipeline():
  '''Make a pipeline for cross-validated grid search for the
      Support Vector Machines Classifier.
  
  This function makes a pipeline which:
      1. Scales the features between 0-1
      2. Selects the KBest features using Anova F-value scoring for
          classification.
      3. Uses KBest features to reduce dimensionality further using PCA
      4. Using the resulting PCA components in Support Vector
          Machines Classifier.
  
  '''
  pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                             ('selection', SelectKBest(score_func=f_classif)),
                             ('reducer', PCA()),
                             ('classifier', SVC())
                             ])
                             
  return pipeline                        

def get_SVC_params(full_search_params=False):
  '''Make a parameters dictionary for cross-validated grid search for the
      Support Vector Machines Classifier.
  
  This function makes a parameter dictionary to search over.
  
  Parameters searched over include:
      SelectKBest:        
          1. k: Number of KBest features to select.
      PCA:
          1. n: Number of PCA components to retain.
          2. whiten: Boolean value whether to whiten the features during PCA.
      SVC:
          1. C: Value of the regularization constraint.
          2. class_weight: Over-/undersamples the samples of each class
          3. tol: Tolerance for stopping criteria
          4. gamma: Kernel coefficient for 'rbf' kernel
          5: kernel: Specifies the kernel type to be used in the algorithm

  Args:
      full_search_params: Boolean value whether to search over an exhaustive 
          grid of params. (Can take a LONG time.)

  Returns:
      A dictionary of parameters to pass into an sk-learn grid-search 
          pipeline. Default parameters include only the final parameters 
          found through exhaustive searching.
  '''
  
  params = {'reducer__n_components': [.5], 
            'reducer__whiten': [True],
            'selection__k':[17],
            'classifier__C': [1],
            'classifier__gamma': [0.0],
            'classifier__kernel': ['rbf'],
            'classifier__tol': [1e-3],
            'classifier__class_weight': ['auto'],
            'classifier__random_state': [42],
            }
            
  if full_search_params:
      params = {'selection__k': [9, 15, 17, 21],
                'classifier__C': [1e-5, 1e-2, 1e-1, 1, 10, 100],
                'classifier__class_weight': [{True: 12, False: 1},
                                             {True: 10, False: 1},
                                             {True: 8, False: 1},
                                             {True: 15, False: 1},
                                             {True: 4, False: 1},
                                             'auto', None],
                'classifier__tol': [1e-1, 1e-2, 1e-4, 1e-8, 1e-16,
                                    1e-32, 1e-64, 1e-128, 1e-256],
                'reducer__n_components': [1, 2, 3, 4, 5, .25, .4, .5, .6,
                                          .75, .9, 'mle'],
                'reducer__whiten': [True, False]
                }
               
  return params
  
from sklearn.feature_selection import SelectPercentile
  
def get_testing_pipeline():
  '''Make a pipeline for cross-validated grid search for a testing model.
  
  This function makes a pipeline which:
      1. Scales the features between 0-1
      2. Selects the KBest features using Anova F-value scoring for
          classification.
      3. Uses KBest features to reduce dimensionality further using PCA
      4. Using the resulting PCA components in Logistic Regression.
  
  '''
  pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                             #('selection', SelectPercentile(score_func=f_classif)),
                             ('reducer', PCA()),
                             ('classifier', SVC())
                             ])
                             
  return pipeline
  
def get_testing_params(full_search_params=False):
  '''Make a parameters dictionary for cross-validated grid search for a
      testing model.
  
  This function makes a parameter dictionary to search over.
  This function is also purely for testing new combinations.

  Args:
      full_search_params: Boolean value whether to search over an exhaustive 
          grid of params. (Can take a LONG time.)

  Returns:
      A dictionary of parameters to pass into an sk-learn grid-search 
          pipeline. Default parameters include only the final parameters 
          found through exhaustive searching.
  '''
  
  params = {'reducer__n_components': [.25, .5],
            'reducer__whiten': [True],
            #'selection__percentile': [10, 15, 20, 25, 30],
            'classifier__C': [1],
            'classifier__gamma': [1e-1],
            'classifier__kernel': ['rbf'],
            'classifier__degree': [3],
            'classifier__tol': [1e-3],
            'classifier__class_weight': [{True: 11, False: 1},
                                         {True: 7, False: 1},
                                         {True: 4, False: 1},
                                         {True: 3, False: 1},
                                         'auto'],
            'classifier__random_state': [42],
            }
            
  if full_search_params:
      params = {'selection__k': [10, 15, 17, 21, 25],
                'classifier__C': [1e-5, 1e-2, 1e-1, 1, 10, 100],
                'classifier__class_weight': [{True: 12, False: 1},
                                             {True: 10, False: 1},
                                             {True: 8, False: 1},
                                             {True: 15, False: 1},
                                             {True: 20, False: 1},
                                             'auto', None],
                'classifier__tol': [1e-1, 1e-2, 1e-4, 1e-8, 1e-16,
                                    1e-32, 1e-64, 1e-128, 1e-256],
                'reducer__n_components': [1, 2, 3, 4, 5, .25, .4, .5, .6,
                                          .75, .9, 'mle'],
                'reducer__whiten': [True, False]
                }
               
  return params