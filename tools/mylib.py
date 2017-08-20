from time import time
from sklearn.metrics import accuracy_score


def fit_and_predict(clf, features_train, features_test, labels_train, labels_test):
    t0 = time()
    clf.fit(features_train, labels_train)
    t1 = time()
    pred = clf.predict(features_test)
    t2 = time()
    accuracy = accuracy_score(labels_test, pred)
    t3 = time()
    print("{:05.3f} : {:05.3f} : {:05.3f} | {:05.3f}".format(t1-t0, t2-t1, t3-t2, t3-t0))
    return accuracy