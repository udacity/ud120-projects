#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    THRESHOLD_ERROR = 80

    ### your code goes here
    for idx, prediction in enumerate(predictions):
        err = prediction - net_worths[idx]
        if abs(err) < THRESHOLD_ERROR:
            cleaned_data.append((ages[idx], net_worths[idx], err))
    print("Cleaned data length: ", len(cleaned_data))
    
    return cleaned_data

