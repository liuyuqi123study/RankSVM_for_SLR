import csv
import os
import pandas as pd
import numpy as np
import json
query=[]
for round in range(5):
    query+=json.load(open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的云端硬盘/LEVEN-main/Downstreams/SCR/SCR-Experiment/input_data/query/query_5fold/query_'+str(round)+'.json'))
cand_path='/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的云端硬盘/LEVEN-main/Downstreams/SCR/SCR-Experiment/input_data/candidates'
for round in range(5):
    csvfile1=pd.read_csv("/Users/yuqi/Downloads/feature_test/features_v1_train_BERT"+str(round)+".csv")
    csvfile2=pd.read_csv('/Users/yuqi/Downloads/feature_test/features_v1_test_BERT'+str(round)+'.csv')
    #我们首先对原来的特征进行读取
    event_id_query_train=pd.DataFrame(np.zeros((len(csvfile1),108)),columns=['q:'+str(i) for i in range(1,109)])
    event_id_cand_train=pd.DataFrame(np.zeros((len(csvfile1),108)),columns=['c:'+str(i) for i in range(1,109)])
    i=0
    
   
    for row in csvfile1.values.tolist():
            qid=str(row[-1])#读取qid
            #print(qid)
            cid=int(row[1][1:-1].split(',')[1].strip().strip("'"))
            for q in query:
               #print(q['ridx'],round)
               if qid==str(q['ridx']):#如果id相同就对应一个query
                   for j in q['inputs']['event_type_ids']:
                       if j!=0:
                       
                        event_id_query_train.loc[i,'q:'+str(j)]=1
                   break
            path=os.path.join(cand_path,os.path.join(qid,str(cid)+'.json'))
            cand=json.load(open(path))#读取candidate
            for k in cand['inputs']['event_type_ids']:
               if k!=0:
                event_id_cand_train.loc[i,'c:'+str(k)]=1
            i=i+1#对每一行进行遍历
    csvfile1=pd.concat((pd.concat((csvfile1,event_id_query_train),axis=1),event_id_cand_train),axis=1)
    csvfile1.to_csv('features_v1_train_BERT_event'+str(round)+'.csv',)
    
    event_id_query_test=pd.DataFrame(np.zeros((len(csvfile2),108)),columns=['q:'+str(i) for i in range(1,109)])
    event_id_cand_test=pd.DataFrame(np.zeros((len(csvfile2),108)),columns=['c:'+str(i) for i in range(1,109)])
    i=0
    for row in csvfile2.values.tolist():
            qid=str(row[-1])
            cid=int(row[1][1:-1].split(',')[1].strip().strip("'"))
            for q in query:
               if qid==str(q['ridx']):
                   for j in q['inputs']['event_type_ids']:
                       if j!=0:
                        event_id_query_test.loc[i,'q:'+str(j)]=1
                   break
            path=os.path.join(cand_path,os.path.join(qid,str(cid)+'.json'))
            cand=json.load(open(path))
            for k in cand['inputs']['event_type_ids']:
               if k!=0:
                event_id_cand_test.loc[i,'c:'+str(k)]=1
            i=i+1
    csvfile2=pd.concat((pd.concat((csvfile2,event_id_query_test),axis=1),event_id_cand_test),axis=1)
    csvfile2.to_csv('features_v1_test_BERT_event'+str(round)+'.csv',)
    
            
            

                        

            

