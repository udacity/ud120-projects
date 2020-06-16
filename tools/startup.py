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
print "downloading the Enron dataset (this may take a while)"
print "to check on progress, you can cd up one level, then execute <ls -lthr>"
print "Enron dataset should be last item on the list, along with its current size"
print "download will complete at about 423 MB"
import urllib
url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"
filename = "../enron_mail_20150507.tar.gz"
try: 
    urllib.urlretrieve(url, filename=filename)
except IOError as socket_error:
    expected_error = (
        "IOError('socket error', SSLError(1, u'[SSL: DH_KEY_TOO_SMALL]"+
        " dh key too small (_ssl.c:727)'))"
        )
    if repr(socket_error) == expected_error:
        import ssl
        cipher = "ECDHE-RSA-AES128-GCM-SHA256"
        context = ssl.create_default_context()
        context.set_ciphers(cipher)
        urllib.urlretrieve(url, filename=filename, context=context)
    else:
        raise socket_error
print "download complete!"


print
print "unzipping Enron dataset (this may take a while)"
import tarfile
import os
os.chdir("..")
tfile = tarfile.open("enron_mail_20150507.tar.gz", "r:gz")
tfile.extractall(".")

print "you're ready to go!"
