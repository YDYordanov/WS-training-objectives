"""Convert the data to the right format"""

import json
import csv
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Transformers model")
parser.add_argument("--data_file", type=str, required=True,
                    help='The prefix of the data file to be processed '
                         '(with .jsonl suffix)')
args = parser.parse_args()

with open(args.data_file+'.jsonl', 'r') as f:
    data = list(f)    
    
label1 = 0
label2 = 0

with open(args.data_file+'.csv', 'w') as out_file:
    writer = csv.writer(out_file, delimiter='\t')
    for item in tqdm(data):
        item = json.loads(item)
        row = []
        label = int(item['answer'])

        if label == 1:
            label = 0
            label1 += 1
        elif label == 2:
            label = 1
            label2 += 1
        else:
            print(label)
            print(item)
            raise ValueError
        
        row.append(label)
        row.append(item['sentence'])
        row.append(item['option1'])
        row.append(item['option2'])
        writer.writerow(row)

print('Data points per label:', label1, label2)
