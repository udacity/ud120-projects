
def add_poi_ratio(data_dict, features_list):
    
    values = ['from_poi_to_this_person', 
              'from_this_person_to_poi',
              'to_messages', 
              'from_messages']
    
    for record in data_dict:
        person = data_dict[record]
        valid = True
        for value in values:
            if person[value] == 'NaN':
                valid = False
                
        if valid:
            poi = person['from_poi_to_this_person'] +\
                  person['from_this_person_to_poi']
            total = person['to_messages'] +\
                    person['from_messages']

            person['poi_ratio'] = float(poi) / total
        else:
            person['poi_ratio'] = 'NaN'
    features_list += ['poi_ratio']