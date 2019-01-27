import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
from sklearn import tree
t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split = 40)
clf = clf.fit(features_train, labels_train)
print "Training time:",round(time()-t0,3),"s"

t1 = time()
pred = clf.predict(features_test)
print "Testing time:", round(time()-t1,3),"s"


from sklearn.metrics import accuracy_score
acc_split_40 = accuracy_score(pred,labels_test)
print "Accuracy is equal to:", (round(acc_split_40,5)*100), "%"
