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

    residual_errors = abs(predictions - net_worths)
    sorted_errors = sorted(residual_errors)

    # Find largest error value that will be allowed
    error_threshold = sorted_errors[int(0.9 * len(residual_errors) - 1)][0]
    print "Error threshold:", error_threshold

    for index in range(0, len(residual_errors)):
        if residual_errors[index][0] <= error_threshold:
            cleaned_data.append((ages[index][0], net_worths[index][0], residual_errors[index][0]))

    return cleaned_data

