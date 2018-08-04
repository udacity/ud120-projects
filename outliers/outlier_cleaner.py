#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    

    ### your code goes here
    
    cleaned_data = [(ages[i], net_worths[i], predictions[i] - net_worths[i]) for i in range(len(ages))]

    cleaned_data.sort(key=lambda point: point[2] * point[2])

    for i in range(10):
        del cleaned_data[-1]



    return cleaned_data

