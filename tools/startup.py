#!/usr/bin/python

print ("\nchecking for nltk")
try:
    import nltk
except ImportError:
    print ("you should install nltk before continuing")

print ("checking for numpy")
try:
    import numpy
except ImportError:
    print ("you should install numpy before continuing")

print ("checking for scipy")
try:
    import scipy
except:
    print ("you should install scipy before continuing")

print ("checking for sklearn")
try:
    import sklearn
except:
    print ("you should install sklearn before continuing")

print ("\ndownloading the Enron dataset (this may take a while)")
print ("to check on progress, you can cd up one level, then execute <ls -lthr>")
print ("Enron dataset should be last item on the list, along with its current size")
print ("download will complete at about 423 MB")

# urlretrieve has been removed in python3, adding correct dependency
# https://stackoverflow.com/questions/17960942/attributeerror-module-object-has-no-attribute-urlretrieve
import urllib.request

url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz"
urllib.request.urlretrieve(url, filename="../enron_mail_20150507.tgz") 
print ("download complete!")

print ("\nunzipping Enron dataset (this may take a while)")
import tarfile
import os
os.chdir("..")
tfile = tarfile.open("enron_mail_20150507.tgz", "r:gz")
tfile.extractall(".")

print ("you're ready to go!")
