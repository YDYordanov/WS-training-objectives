"""
This script reformats the DPR data into WinoGrande input format
This script requires the Kaggle version of the DPR dataset (with pronoun locations).
Note: DPR-test is named winogrande_train.csv in the Kaggle version.
"""

import csv
import json
    
    
with open('winograd_train.csv') as json_file:
    data = csv.DictReader(json_file, delimiter=',')
    data = list(data)
    
final_data = []

# Now extract the alternatives and reformat the data
for data_id, dat in enumerate(data):
    true_id = (1 if dat['A-coref'] == 'True' else 2)
    sent = dat['Text']
    
    # mask the pronoun:
    pron_location = int(dat['Pronoun-offset'])
    pron_length = len(dat['Pronoun'])
    sent = sent[:pron_location] + '_' + sent[pron_location + pron_length: ]
    
    option1 = dat['A']
    option2 = dat['B']    
    final_data.append({
        "qID": str(data_id), 'sentence': sent, 'option1': option1,
        'option2': option2, 'answer': true_id        
        })

with open('DPR_test.jsonl', 'w') as file_:
    for entry in final_data:
        json.dump(entry, file_)
        file_.write('\n')

