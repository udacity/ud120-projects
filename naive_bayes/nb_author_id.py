#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# Example P[robability](C[ancer]) = 0.01 # Occurs in 1% of the population
# Test: 90% it is positive if you have C[ancer] # Sensitivity
#       90% it is negative if you don't have C  # Specitivity
# Question: Test = Positive; Probability of having cancer

# Prior probability ( P(C) ) * test evidence = Posterior probability

# Prior:        P(C)     = 0.01 = 1%
#               P(!C)    = 0.99 = 99%
#               P(pos|C) = 0.9  = 90%
# Sensitivity:  P(pos|!C)= 0.1  = 10%
# Specitivity:  P(neg|!C)= 0.9  = 90%
#
# Joint Probablity: Multiplizierte Wahrscheinlichkeit
#               P(C,pos) = P(C) * P(pos|C)
#                        = 0.01 * 0.9 
#                        = 0.009
#               P(!C,pos)= P(!C) * P(pos|!C)
#                        = 0.99 * 0.1
#                        = 0.099
# 
# Normalizer:   P(pos)   = P(C,pos) + P(!C,pos) # Summe Joint Probability Ergebnisse
#                        = 0.009 + 0.099 
#                        = 0.108
# 
# Posterior:    P(C|pos) = P(C,pos) / P(pos)
#                        = 0.009 / 0.108
#                        = 0.083333333
#               P(!C|pos)= P(!C,pos) / P(pos)
#                        = 0.099 / 0.108
#                        = 0.916666667
#
# Sum Posterior: P(C|pos) + P(!C|pos) = 0.083333333 + 0.916666667 = 1
#
# Bayes rule: 
#   1. Wahrscheinlichkeit von C
#   2. Wahrscheinlichkeit von C trifft zu ( d.h. ist positiv ) 
#   3. Wahrscheinlichkeit von C trifft nicht zu ( d.h. ist negativ ) 
#   4. Berechne C ( Gesamt ) is positiv mittels: 
#       4.1 Wkeit von C positiv * Wkeit von C
#       4.2 Wkeit von nicht C positiv * Wkeit von C # nicht C = 1 - C
#   5. Berechne Normalizer mittels Summiere 4.1 und 4.2
#   6. Normalisiere 4.1 und 4.2 mittels: 
#       6.1 Wkeit von C positiv / Normalizer
#       6.2 Wkeit von nicht C positiv / Normalizer
#   7. Pruefe: 6.1 und 6.2 muessen 1 ergeben
#########################################################


