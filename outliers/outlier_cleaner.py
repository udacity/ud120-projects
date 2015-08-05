#!/usr/bin/python

import numpy as np


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)
        
        return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error)
        """
    
    cleaned_data = []
    
    err = (predictions - net_worths) ** 2
    
    temp_data = zip(ages, net_worths, err)
    
    temp_data = sorted(temp_data, key = lambda data: data[2])
    
    clean_points = int(0.9 * len(predictions))
    
    cleaned_data = temp_data[:clean_points]
    
    return cleaned_data