import json
import os
cand_path='/Users/yuqi/Downloads/LEVEN-main/Downstreams/SCR/SCR-Preprocess/input_data/data/candidates'
query_path='/Users/yuqi/Downloads/LEVEN-main/Downstreams/SCR/SCR-Preprocess/input_data/data/query'
length_query=[]
length_candidate=[]
for fn in os.listdir(query_path):
    with open(os.path.join(query_path,fn),'r') as f1:
        for line in f1:
            length_query.append(len(json.loads(line)['q']))
for fn in os.listdir(cand_path):
    if fn=='.DS_Store':
        continue
    for f2 in os.listdir(os.path.join(cand_path,fn)):
        with open(os.path.join(cand_path,os.path.join(fn,f2))) as f:
            for line in f:
                length_candidate.append(len(json.loads(line)['ajjbqk']))
import seaborn as sns
import matplotlib.pyplot as plt
ax=sns.histplot(length_query)
ax.set_xlabel('length of fact part of query files')
ax.set_xticks(range(0,1800,200))
plt.show()
ax2=sns.histplot(length_candidate)
ax2.set_xlabel('length of fact part of candidate files')
ax2.set_xticks(range(0,25000,5000))
plt.show()

