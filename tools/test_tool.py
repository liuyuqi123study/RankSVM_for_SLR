from timeit import default_timer as timer
import torch
from torch.autograd import Variable
from timeit import default_timer as timer

def test_RankNet(model,dataset,gpu_list):
    model.eval()

    acc_result=None
    total_loss=0
    total_len=len(dataset)
    start_time=timer()
    output_info=""

    res_scores=[]
    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key],torch.Tensor):
                if len(gpu_list)>0:
                    data[key]=Variable(data[key].cuda())
                else:
                    data[key]=Variable(data[key])
        with torch.no_grad():
            predictions=model(data['inputx'][:,0])
            predictions_2=model(data['inputx'][:,1])



    return