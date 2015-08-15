#!/usr/bin/python

import os
from tarfile import open
from urllib import urlretrieve


realpath = os.path.dirname(os.path.realpath(__file__))

print("")
print("checking for nltk")
try:
    import nltk
except ImportError:
    print("you should install nltk before continuing")
    exit()

print("checking for numpy")
try:
    import numpy
except ImportError:
    print "you should install numpy before continuing"
    exit()

print("checking for sklearn")
try:
    import sklearn
except ImportError:
    print "you should install sklearn before continuing"
    exit()

tar_exists = False

try:
    tar_exists = os.path.isfile(realpath + "/../enron_mail_20150507.tgz")
except IOError:
    pass

if tar_exists:
    print("you already have the Enron tarball")
    print("if you would like a new version, please remove the tarball and run this script again")
else:
    print("")
    print("downloading the Enron dataset (this may take a while)")
    print("to check on progress, execute <ls -lthr " + realpath + "/..>")
    print("Enron dataset should be last item on the list, along with its current size")
    print("download will complete at about 423 MB")
    url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz"
    urlretrieve(url, filename=realpath+"/../enron_mail_20150507.tgz")
    print("download complete!")

try:
    maildir_exists = os.path.isdir(realpath + "/../maildir")
except IOError:
    pass

if maildir_exists:
    print("you already have the unzipped Enron emails. Check maildir")
    print("if you would like a new version, please remove the maildir directory and run this script again")
else:
    print("")
    print("unzipping Enron dataset (this may take a while)")
    tfile = open(realpath + "/../enron_mail_20150507.tgz", "r:gz")
    tfile.extractall(realpath+"/..")

print("you're ready to go!")
