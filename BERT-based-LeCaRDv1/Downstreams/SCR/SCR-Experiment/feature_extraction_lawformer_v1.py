import argparse
import os
import torch
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

parser = argparse.ArgumentParser()#在这里进行参数的传递
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
#config.set('distributed', 'local_rank', args.local_rank)
#if config.getboolean("distributed", "use"):
#        torch.cuda.set_device(gpu_list[args.local_rank])
 #       torch.distributed.init_process_group(backend=config.get("distributed", "backend"))
 #       config.set('distributed', 'gpu_num', len(gpu_list))

logger = logging.getLogger(__name__)
logger.info("Begin to initialize models...")

model = get_model(config.get("model", "model_name"))(config, gpu_list,  local_rank = -1)

checkpoint="/content/input/model/4.pkl"
if len(gpu_list) > 0:
    
        if args.local_rank<0:
            model = model.cuda()
        else:
            model = model.to(gpu_list[args.local_rank])

try:
        parameters = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        if hasattr(model, 'module'):
            model.module.load_state_dict(parameters["model"])
        else:
            model.load_state_dict(parameters["model"])
except Exception as e:
        information = "Cannot load checkpoint file with error %s" % str(e)
        logger.warning(information)
if torch.backends.mps.is_available():
                           model=model.to('mps')
model.cuda()

activation={}
def get_activation(name):
      def hook(model,input,output):
            activation[name]=output.pooler_output.detach()
      return hook


from tools.init_tool import init_all
para = init_all(config, gpu_list, checkpoint, "train", local_rank = args.local_rank)
model.encoder.register_forward_hook(get_activation('encoder'))

dataset=para['train_dataset']
acc_result = {'right': 0, 'actual_num': 0, 'pre_num': 0}
for step, data in enumerate(dataset):
    activation={}
    for key in data.keys():
        if key!='index' and data[key]!=None:
         if torch.backends.mps.is_available():
                                data[key] = Variable(data[key].to('mps'))
         else:
            
            if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
            else:
                        data[key] = Variable(data[key])
    with torch.no_grad():
        output=model(data,config,gpu_list,acc_result,'train')
    activation['encoder']=activation['encoder'].tolist()
    activation['label']=data['labels'].tolist()
    activation['id']=data['index']
    import json
    fout = open('/content/features/v1_feature_extraction_lawformer_train_epoch4_testfile2'+str(step)+'.json', "w")
    print(json.dumps(activation), file = fout)
dataset=para['valid_dataset']
acc_result = {'right': 0, 'actual_num': 0, 'pre_num': 0}
for step, data in enumerate(dataset):
    print(step)
    activation={}
    for key in data.keys():
        if key!='index' and data[key]!=None:
            if torch.backends.mps.is_available():
                                data[key] = Variable(data[key].to('mps'))
            else:
                 if len(gpu_list) > 0:
                       data[key] = Variable(data[key].cuda())
                 else:
                       data[key] = Variable(data[key])
    with torch.no_grad():
        output=model(data,config,gpu_list,acc_result,'train')
    activation['encoder']=activation['encoder'].tolist()
    activation['label']=data['labels'].tolist()
    activation['id']=data['index']
    import json
    fout = open('/content/features/lawformer_v1_feature_extraction_test_epoch4_testfile2'+str(step)+'.json', "w")
    print(json.dumps(activation), file = fout)
