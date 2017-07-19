#!/usr/bin/python

from sys import stdout

def reporthook(download_description):
    """Build a reporthook for retrieving with urllib.

    Args:
        download_description (str): a description of the download
        that will be displayed at any downloaded HTTP chunk.
    """
    spinner_frames = ['|', '/', '-', '\\']

    def hook(chunk_number, chunk_size, download_size):
        if chunk_size < download_size:
            hook.received_size += chunk_size
        else:
            hook.received_size = download_size

        progress_percentage = int(100*hook.received_size/download_size)
        spinner_frame = spinner_frames[chunk_number % len(spinner_frames)]

        stdout.write("downloading {} ({}MB): {:2d}% {}\r".format(
            download_description, int(download_size/1e6), progress_percentage, spinner_frame))
        stdout.flush()

    hook.received_size = 0
    return hook

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

print "checking for scipy"
try:
    import scipy
except:
    print "you should install scipy before continuing"

print "checking for sklearn"
try:
    import sklearn
except:
    print "you should install sklearn before continuing"

print
import urllib
url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz"
urllib.urlretrieve(url, filename="../enron_mail_20150507.tgz", reporthook=reporthook("Enron dataset"))
print
print "download complete"

print
print "unzipping Enron dataset (this may take a while)"
import tarfile
import os
os.chdir("..")
tfile = tarfile.open("enron_mail_20150507.tgz", "r:gz")
tfile.extractall(".")

print "you're ready to go!"
