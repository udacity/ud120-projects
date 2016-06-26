#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    # print(predictions)
    # print(ages)
    # print(net_worths)

    cleaned_data = []

    pre_cleaned_data = []

    i = 0

    for prediction in predictions:
        error = (net_worths[i] - prediction) ** 2
        data = (ages[i], net_worths[i], error)
        pre_cleaned_data.append(data)
        i += 1

    pre_cleaned_data = sorted(pre_cleaned_data, key=lambda x: x[2])

    datalength = pre_cleaned_data.__len__()
    stoppoint = datalength * .9
    i = 0

    while i < stoppoint:
        cleaned_data.append(pre_cleaned_data[i])
        i += 1

    ### your code goes here


    return cleaned_data

