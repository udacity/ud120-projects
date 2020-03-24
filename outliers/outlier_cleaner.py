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
    agelist = [i[0] for i in ages]
    net_worthslist = [i[0] for i in net_worths]
    errorlist =[i[0] for i in abs(net_worths-predictions)]
    cleaned_data = zip(agelist,net_worthslist,errorlist)
    ### funtion for sorting by errors
    #def sort3key(ele):
        #return ele[2]
    cleaned_data.sort(key= lambda ele: ele[2])
    ### chop off 10% largest error values
    cleaned_data =cleaned_data[:int(len(cleaned_data)*.9)]
    return cleaned_data
