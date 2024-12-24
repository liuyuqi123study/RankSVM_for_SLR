from sklearn.model_selection import KFold
import json
folds=KFold(n_splits=5,shuffle=True,random_state=None)
querys=[]
with open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/CAIL2024/LEVEN-main-2/Downstreams/SCR/SCR-Preprocess/output_data-supervised/query/whole_query.json') as files:
        querys=json.loads(files.read())
for fold_i,(train_index,val_index) in enumerate(folds.split(querys)):
    query_list=querys[val_index]
    with open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/CAIL2024/LEVEN-main-2/Downstreams/SCR/SCR-Preprocess/output_data-supervised/query/'+str(fold_i)+'.json') as f:
        json_data=json.dumps(query_list)
        f.write(json_data)