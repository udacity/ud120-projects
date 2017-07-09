#!/usr/bin/python

import os
import tarfile
import urllib

print("\nchecking for nltk")
try:
    import nltk
except ImportError:
    print("you should install nltk before continuing")

print("checking for numpy")
try:
    import numpy
except ImportError:
    print("you should install numpy before continuing")

print("checking for scipy")
try:
    import scipy
except:
    print("you should install scipy before continuing")

print("checking for sklearn")
try:
    import sklearn
except:
    print("you should install sklearn before continuing")

print("""\ndownloading the Enron dataset (this may take a while)
to check on progress, you can cd up one level, then execute <ls -lthr>
Enron dataset should be last item on the list, along with its current size
download will complete at about 423 MB""")

url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz"
urllib.urlretrieve(url, filename="../enron_mail_20150507.tgz") 
print("download complete!")

print("\nunzipping Enron dataset (this may take a while)")

os.chdir("..")
tfile = tarfile.open("enron_mail_20150507.tgz", mode="r:gz")
tfile.extractall(".")
tfile.close()

print("you're ready to go!")
