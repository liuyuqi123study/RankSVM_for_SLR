import json
results= json.load(open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/LEVEN-main/Downstreams/SCR/SCR-Experiment/result/EDBERT/test/PairwiseLecardBert-test-0_epoch-1.json', 'r'))
qid='-5180'

rank_file=results[qid]
query_file=[]
with open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/LEVEN-main/Downstreams/SCR/SCR-Experiment/input_data/query/query.json','r') as f:
    for line in f:
        query_file.append(json.loads(line))
for query in query_file:
    if query['ridx']==-5180:
        print(query['q'][:109])
        break
length=10
print('这是候选案件')
for i in range(length):
    file_name='/Users/yuqi/Downloads/LEVEN-main/Downstreams/SCR/SCR-Experiment/input_data/candidates/'+str(-5180)+'/'+str(rank_file[i])+'.json'
    cand=json.load(open(file_name,'r'))
    print(cand['ajjbqk'][:400])
    #print(i)
    label=json.load(open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/LEVEN-main/Downstreams/SCR/SCR-Experiment/input_data/label/label_top30_dict.json','r'))
    print(label['-5180'][str(rank_file[i])])
    print(str(rank_file[i]))

