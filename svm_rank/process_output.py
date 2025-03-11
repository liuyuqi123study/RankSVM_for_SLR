#Here we need to read the output of RankSVM model
#So we need to build dictionaries
#We need to first read them
import json
for c in [0.005,0.05,0.02,0.01,0.1,0.5,1]:
    for i in range(5):
        results={}
        with open('predictions_'+str(i)+'_'+str(c)+'.dat', 'r') as file:
            predictions = file.readlines()

        with open('test_feature_modified'+str(i)+'.dat', 'r') as file:
            test_features = file.readlines()
        for j in range(len(predictions)):
            qid=int(test_features[j].split(' ')[1].replace('qid:',''))-5181
    
            cid=test_features[j].split('#')[1].strip('\n').strip('"')
            
            if qid not in results.keys():
                results[qid]={}
            
            results[qid][cid]=float(predictions[j])
        for qid in results.keys():
            results[qid]=sorted(results[qid],key=lambda k:results[qid][k],reverse=True)
        with open('prediction_mamba_ranksvm_'+str(c)+'_'+str(i)+'.json','w') as json_file:
            json.dump(results,json_file)
        
        
    
    
        