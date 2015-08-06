# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:41:26 2015

@author: jayantsahewal
"""

from nltk.corpus import stopwords

sw = stopwords.words("english")

print len(sw)

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")