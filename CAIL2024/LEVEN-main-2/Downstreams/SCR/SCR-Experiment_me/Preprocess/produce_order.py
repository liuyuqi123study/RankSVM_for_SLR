import json
label={}
for line in open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/CAIL2024/relevence.trec'):
  data=line.split('\t')
  query_name=str(data[0])
  if query_name not in label.keys():
    label[query_name]={}
  label[query_name][str(data[2])]=int(data[3][:-1])
with open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/CAIL2024/LEVEN-main-2/Downstreams/SCR/SCR-Experiment_me/input_data/label/label_order.json','w') as f:
                f.write(json.dumps(label,ensure_ascii=False))