import argparse
import os
import torch
import json
import logging
#import torchvision
from torch.autograd import Variable
import torchvision.models.feature_extraction

#from tools.init_tool import init_all
from config_parser import create_config
#from tools.train_tool import train
import numpy as np
from model import get_model
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()#在这里进行参数的传递

    parser.add_argument
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    parser.add_argument('--do_test', help="do test while training or not", action="store_true")
    parser.add_argument('--comment', help="checkpoint file path", default=None)
    parser.add_argument('--test_file', default=None)
    parser.add_argument("--seed", default=2333)#这些都是可选的参数
    args = parser.parse_args()

    configFilePath = args.config

    config = create_config(configFilePath)
        
    if not args.test_file is None:
            config.set("data", "test_file", args.test_file)
    set_seed(args.seed)

    use_gpu = True
    gpu_list = []
    if args.gpu is None:
            use_gpu = False
    else:
            use_gpu = True
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

            device_list = args.gpu.split(",")
            for a in range(0, len(device_list)):
                gpu_list.append(int(a))

    os.system("clear")
    # We just need to load the dataset

    from tools.init_tool import init_all
    #First step: Initializing the dataset
    para = init_all(config, gpu_list, checkpoint=None, mode="train", local_rank = args.local_rank)
    dataset=para['train_dataset']
    acc_result = {'right': 0, 'actual_num': 0, 'pre_num': 0}
    
    #We need to load features from .json file
    #second step: load the feature file
    train_features=[]
    train_labels=[]
    train_indexes=[]

    test_features=[]
    test_labels=[]
    test_indexes=[]

    repository_path=config.get("data","feature_path")
    for file_name in os.listdir(repository_path):
        if file_name.endswith('.json'):
            
            train_or_test=file_name.split('_')[3]
            if train_or_test=='train':
                  step=file_name.split('_')[5].replace('.json','').replace('testfile1','')
                  if int(step)>=100:
                        continue
                  file_path=os.path.join(repository_path,file_name)
                  with open(file_path,'r') as f:
                    data=json.load(f)
                    features=data['encoder']
                    labels=data['label']
                    indexes=data['id']

                    train_features.append(features)
                    train_labels.append(labels)
                    train_indexes.append(indexes)

            else:
                  step=file_path.split('_')[5].replace('.json','').replace('testfile1','')
                  if int(step)>=25:
                        continue
                  file_path=os.path.join(repository_path,file_name)
                  with open(file_path,'r') as f:
                    data=json.load(f)
                    features=data['encoder']
                    labels=data['label']
                    indexes=data['id']

                    test_features.append(features)
                    test_labels.append(labels)
                    test_indexes.append(indexes)
       
            