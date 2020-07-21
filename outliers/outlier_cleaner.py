#!/usr/bin/python
import math
import numpy


def outlier_cleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
    errors = net_worths - predictions
    threshold = numpy.percentile(numpy.absolute(errors), 90)
    cleaned_data = [
        (age, net_worth, error)
        for age, net_worth, error in zip(ages, net_worths, errors)
        if abs(error) <= threshold
    ]
    return cleaned_data
