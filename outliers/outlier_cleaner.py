#!/usr/bin/python

import pandas as pd
import math

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
   # errors = zip(predictions,net_worths)
    
    result = zip(predictions,ages,net_worths)
    
    result_df = pd.DataFrame(result, columns = ['predictions', 'ages', 'net_worths'])
    
    result_df['errors'] = math.fabs(result_df['net_worths'] - result_df['predictions'])
    
    result_sorted = result_df.sort_values(by = 'errors', ascending = True)
    
    result_sorted.head(80)
    
    print "what is going on?"
    
    cleaned_data = zip(result_sorted['ages'], result_sorted['net_worths'], result_sorted['errors'])
    
       

    ### your code goes here

    
    return cleaned_data

