import json
import os
import pandas as pd
csvfile=pd.read_csv('/Users/yuqi/Downloads/feature_test/features_v2_train_130.csv')
i=0
art=[]
for line in csvfile.values.tolist():
    cand_id=str(line[1][1:-1].split(',')[1].strip().strip("'"))
    with open(os.path.join('/Users/yuqi/Downloads/candidate_55192',str(cand_id)+'.json')) as f:
        cand=json.load(f)
        articles=cand['article']
        for a in articles:
            if 'a:'+str(a) not in csvfile.columns:
                csvfile.loc[:,'a:'+str(a)]=0
                art.append(a)
            csvfile.loc[i,'a:'+str(a)]=1
        i=i+1
csvfile.to_csv('/Users/yuqi/Downloads/feature_test/features_v2_train_130_articles.csv',index=False,header=True)
print(art)