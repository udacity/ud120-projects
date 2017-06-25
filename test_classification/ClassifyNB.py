def classify(features_train, labels_train):
    from sklearn import svm
    clf = svm.SVC(kernel = 'rbf', C = 1, gamma = 1000)
    return clf.fit(features_train, labels_train)


    ### your code goes here!
