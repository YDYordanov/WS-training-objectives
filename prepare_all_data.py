"""
Prepare WG, WG-dev, WSC, WSC-modified and DPR(test) datasets
"""

import os


for dataset in ['WinoGrande/train_xl', 'WinoGrande/dev', 'WSC/WSC',
                'WSC/WSC_modified', 'DPR/DPR_test']:
    print('Preparing', dataset)
    os.system('python Data/prep_data.py --data_file=Data/{}'.format(dataset))

# Send the datasets to WG-SR:
print('Copying data to WG-SR')
os.system('cp Data/WinoGrande/train_xl.jsonl winogrande/Data/train.jsonl')
os.system('cp Data/WinoGrande/dev.jsonl winogrande/Data/dev.jsonl')
for dataset in ['WinoGrande/dev', 'WSC/WSC', 'DPR/DPR_test']:
    folder = dataset[: dataset.find('/')]
    os.system('cp Data/{}.jsonl winogrande/Data/{}/dev.jsonl'.format(
        dataset, folder))
print('Done!')

