from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from classify import NBAccuracy
import sklearn

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy
    
if __name__ == '__main__':
    print submitAccuracy()