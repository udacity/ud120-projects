#!/usr/bin/python

print
print "checking for nltk"
try:
    import nltk
except ImportError:
    print "you should install nltk before continuing"

print "checking for numpy"
try:
    import numpy
except ImportError:
    print "you should install numpy before continuing"

print "checking for sklearn"
try:
    import sklearn
except:
    print "you should install sklearn before continuing"

print
print "downloading the Enron dataset (this may take a while)"
print "to check on progress, you can cd up one level, then execute <ls -lthr>"
print "Enron dataset should be last item on the list, along with its current size"
print "download will complete at about 423 MB"
import urllib
url = "https://www.cs.cmu.edu/~./enron/enron_mail_20110402.tgz"
import sys

def dlProgress(count, blockSize, totalSize):
      percent = int(count*blockSize*100.0/totalSize)
      sys.stdout.write("\r" +"File :" + "...%d%%  --- %f kb" % (percent,count*blockSize/1000.0))
      sys.stdout.flush()


urllib.urlretrieve(url, filename="../enron_mail_20110402.tgz",reporthook=dlProgress)

print "download complete!"


print
print "unzipping Enron dataset (this may take a while)"
import tarfile
import os
os.chdir("..")
tfile = tarfile.open("enron_mail_20110402.tgz", "r:gz")
tfile.extractall(".")

print "you're ready to go!"
