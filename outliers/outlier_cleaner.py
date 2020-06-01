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
    import math
    num_values_to_remove = int(len(predictions) * .1)

    errors_list = []
    for i in range(len(predictions)):
        errors_list.extend(abs(predictions[i] - net_worths[i]))

    errors_list.sort()
    top_errors = errors_list[len(predictions) - num_values_to_remove:]

    for i in range(len(predictions)):
        error = abs(predictions[i] - net_worths[i])
        if error in top_errors:
            pass
        else:
            cleaned_data.append([ages[i],net_worths[i],error])
    print(cleaned_data)

    return cleaned_data

