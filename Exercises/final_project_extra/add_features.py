# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:07:16 2015

@author: jayantsahewal
"""

def add_totals(data_dict, features_list):
    """ Mutates data dict to add total compensation and total poi interaction """
    """ Finally returns the updated data dictionary and features """
    
    """ add total compensation """
    
    data_dict = data_dict
    features_list = features_list
    
    financial_fields = ['total_stock_value', 'total_payments']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in financial_fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            person['total_compensation'] = sum([person[field] for field in financial_fields])
        else:
            person['total_compensation'] = 'NaN'
    features_list += ['total_compensation']

    """ add total poi interaction """
    
    email_fields = ['shared_receipt_with_poi', 'from_this_person_to_poi', 'from_poi_to_this_person']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in email_fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            person['total_poi_interaction'] = sum([person[field] for field in email_fields])
        else:
            person['total_poi_interaction'] = 'NaN'
    features_list += ['total_poi_interaction']
    
    return data_dict, features_list