import json
labels={}
for line in open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/CAIL2024/relevence_gold.trec'):
  data=line.split('\t')
  folder_name=data[0]
  if folder_name not in labels.keys():
    labels[str(folder_name)]=[]
  labels[str(folder_name)].append(str(data[2]))
with open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/CAIL2024/LEVEN-main-2/Downstreams/SCR/SCR-Experiment_me/input_data/label/labels_train_gold.json','w') as f:
                f.write(json.dumps(labels,ensure_ascii=False))