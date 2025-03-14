import torch
import logging
from torch.autograd import Variable
logger=logging.getLogger
from timeit import default_timer as timer
from tools.test_tool import test_RankNet

def ranknet_loss(s_i, s_j, y_ij):
    """
    s_i: Score for document i
    s_j: Score for document j
    y_ij: Ground truth label (1 if i > j, -1 if i < j, 0 if i == j)
    """
    s_diff = s_i - s_j
    loss = torch.log(1 + torch.exp(-y_ij * s_diff))
    return loss.mean()

def train_ranknet(parameters,config,gpu_list):

    epochs=config.getint("train","epoch")
    model=parameters['model']
    optimizer=parameters['optimizer']
    dataset=parameters['train_dataset']
    model.train()
    for epoch in range(epochs):
        start_time=timer()
        acc_result=None
        epoch_loss = 0

        for step,data in dataset:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])
            
            optimizer.zero_grad()
            results=model(data,config,gpu_list,acc_result,"train")
            loss,acc_result=results['loss'],results['acc_result']

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(data)}")
    #We need to write codes for test data
    with torch.no_grad():
        test_RankNet(model,parameters['valid_dataset'],gpu_list)