import os
import csv
dataset={}
for folder in os.listdir('/Users/yuqi/Downloads/LEVEN-main/Downstreams/SCR/SCR-Preprocess/input_data/data/candidates'):
    if folder=='.DS_Store':
        continue
    for fn in os.listdir(os.path.join('/Users/yuqi/Downloads/LEVEN-main/Downstreams/SCR/SCR-Preprocess/input_data/data/candidates',folder)):
        if fn=='.DS_Store':
            continue
        if folder not in dataset:
            dataset[folder]=[fn.replace('.json','')]
        else:
            dataset[folder].append(fn.replace('.json',''))
with open('dataset.csv','w',newline='') as f:
    writer=csv.writer(f)
    for row in dataset.items():
        writer.writerow(row)