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
    

    result_df['errors'] = (result_df['net_worths'] - result_df['predictions']).abs()
    
    result_df = result_df.sort_values(by = 'errors', ascending = True)
    
    result_df = result_df.head(80)

    #result_df['errors'] = math.fabs(result_df['net_worths'] - result_df['predictions'])
    
    #result_sorted = result_df.sort_values(by = 'errors', ascending = True)
    
    #result_sorted.head(80)
    
    results_out = result_df.head(80)

    
    #print result_df.head(5)

   # print "what is going on?"
=======
    cleaned_data = zip(results_out['ages'], results_out['net_worths'], results_out['predictions'])
>>>>>>> Stashed changes
    
    cleaned_data = zip(result_df['ages'], result_df['net_worths'], result_df['errors'])
    
   # print cleaned_data[0:5]
    
  

    ### your code goes here

    
    return cleaned_data

