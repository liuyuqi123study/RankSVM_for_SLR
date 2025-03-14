from timeit import default_timer as timer
import torch
from torch.autograd import Variable
import os
import json

def test_RankNet(model,dataset,config,gpu_list):
    model.eval()

    acc_result=None
    total_loss=0
    total_len=len(dataset)
    start_time=timer()
    output_info=""


    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key],torch.Tensor):
                if len(gpu_list)>0:
                    data[key]=Variable(data[key].cuda())
                else:
                    data[key]=Variable(data[key])
        with torch.no_grad():
            predictions=model(data,config,gpu_list,acc_result,'valid')
        loss,acc_result=predictions["loss"],predictions["acc_result"]
        total_loss+=float(loss)
        res_scores+=list(zip(predictions["index"][:,0],predictions["score_1"]))
        res_scores_2+=list(zip(predictions["index"][:,1],predictions["score_2"]))
    
    del data
    del results

    predictions={}
    for res in res_scores:
        if res[0][0] not in predictions:
            predictions[res[0][0]]=[]
        predictions[res[0][0]].append((res[0][1],res[1]))

    for key in predictions:
        predictions[key].sort(key=lambda x:x[1],reverse=True)
        predictions[key]=[int(res[0] for res in predictions[key])]

    os.makedirs(config.get("data","result_path"),exist_ok=True)
    fout=open(os.path.join(config.get("data","result_path"),"%s-test-%d_epoch-%d.json"%(config.get("output","model_name"),config.getint("data","fold"),epoch)),'w')
    print(json.dumps(predictions),file=fout)
    fout.close()

    return