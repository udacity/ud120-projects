#!/usr/bin/python
import math

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    # def calculateError(index):
    #     pred = predictions[index][0]
    #     netWorth = net_worths[index][0]
    #     age = ages[index][0]
    #     error = (netWorth - pred)**2
    #     return (age, netWorth, error)
    #
    # setSize = len(predictions)
    #
    # errors = map(calculateError, range(0, setSize))
    # errors.sort()
    #
    #
    #
    # cleaned_data = errors[0:int(0.9 * setSize)]
    # print cleaned_data
    # return cleaned_data
    errors = (net_worths-predictions)**2
    cleaned_data =zip(ages,net_worths,errors)
    cleaned_data = sorted(cleaned_data,key=lambda x:x[2][0], reverse=True)
    limit = int(len(net_worths)*0.1)
    return cleaned_data[limit:]
