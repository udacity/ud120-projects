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

    ### your code goes here
    import operator

    errors = [a - b for a, b in zip(predictions, net_worths)]
    data = zip(ages, net_worths, errors)
    data.sort(key=operator.itemgetter(2))
    cleaned_data = data[:int(len(predictions) * 0.9)]
    
    return cleaned_data

