#!/usr/bin/python3

print("Checking for nltk")
try:
    import nltk
except ImportError:
    print("You should install nltk before continuing")

print("Checking for numpy")
try:
    import numpy
except ImportError:
    print("You should install numpy before continuing")

print("Checking for scipy")
try:
    import scipy
except:
    print("You should install scipy before continuing")

print("Checking for sklearn")
try:
    import sklearn
except:
    print("You should install sklearn before continuing")

print("Downloading the Enron dataset (this may take a while)")
print("To check on progress, you can cd up one level, then execute <ls -lthr>")
print("Enron dataset should be last item on the list, along with its current size")
print("Download will complete at about 1.82 GB")

import requests
url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"
filename = "../enron_mail_20150507.tar.gz"
with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)
print("Download Complete!")

print("Unzipping Enron dataset (This may take a while)")
import tarfile
tfile = tarfile.open("../enron_mail_20150507.tar.gz")
tfile.extractall(".")
tfile.close()

print("You're ready to go!")

