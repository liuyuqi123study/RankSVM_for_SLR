#Here we need to read the output of RankSVM model
#So we need to build dictionaries
#We need to first read them
import json
for i in range(5):
    results={}
    with open('predictions_'+str(i)+'.dat', 'r') as file:
        predictions = file.readlines()

    with open('test_feature_modified'+str(i)+'.dat', 'r') as file:
        test_features = file.readlines()
    for j in range(len(predictions)):
        qid=int(test_features[j].split(' ')[1].replace('qid:',''))-5181
  
        cid=test_features[j].split('#')[1].strip('\n').strip('"')
        print(cid)
        if qid not in results.keys():
            results[qid]={}
        
        results[qid][cid]=float(predictions[j])
    for qid in results.keys():
        results[qid]=sorted(results[qid],key=lambda k:results[qid][k],reverse=True)
    with open('prediction_mamba_ranksvm_'+str(i)+'.json','w') as json_file:
        json.dump(results,json_file)
    
    
    
    
        