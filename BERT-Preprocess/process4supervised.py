from transformers import BertTokenizer
from bert_crf import BertCRFForTokenClassification,BertConfig
from transformers import AutoTokenizer,AutoModelForMaskedLM
import torch#在这里使用的是torch
from tqdm import tqdm
import jsonlines
import os
import json
import warnings
import random
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#天呐，不要直接修改，做了注释再修改

class BertCrfTagging(object):#这里使用了一个叫做BertCrfTagging的模型
    def __init__(self,
                 data_path='./Downstreams/SCR/SCR-Preprocess/input_data/data/query/query.json',
                 save_path='./Downstreams/SCR/SCR-Preprocess/output_data/query/query.json',
                 checkpoint='./Downstreams/SCR/SCR-Preprocess/saved/checkpoint-1900',
                 max_length=100,
                 batch_size=16,
                 device=torch.device('cpu'),
                 mode='query'):
        self.device = device
      #命名实体识别任务的主要工作是识别出比如说姚明和NBA两个实体
        self.model = BertCRFForTokenClassification.from_pretrained(checkpoint)#这是一个命名实体识别的模型，checkpoint的作用是保存模型参数，所以是不是关键在于它对checkpoint进行了修改
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)#使用的预训练模型是bert-base-chinese，和清华训练好的checkpoint全部放在了一个文件夹下哈哈
        #self.tokenizer=AutoTokenizer.from_pretrained("Downstreams/SCR/SCR-Preprocess/bert-base-chinese")
        #self.model=AutoModelForMaskedLM.from_pretrained("Downstreams/SCR/SCR-Preprocess/bert-base-chinese",config=self.tokenizer)
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.save_path = save_path
        self.mode = mode
        print('using {} device'.format(device))

    def process(self):
        if self.mode == 'query':
            data = list(jsonlines.open(self.data_path))#这里使用LeCaDRD的数据，data_path中包含的事query的基本信息
            data = self.generate_batch(data)
            with jsonlines.open(os.path.join(self.save_path), 'w') as f:#将代码的输出写到query.json文件中
                for i, batch in enumerate(tqdm(data, desc='writing {} file'.format(self.mode))):
                    data[i] = self.add_inputs(batch)#对数据按批存入并进行编号
                    for d in data[i]:
                        jsonlines.Writer.write(f, d)
        else:
            for folder in tqdm(os.listdir(self.data_path), desc='writing {}'.format(self.mode)):
                os.makedirs(os.path.join(self.save_path, folder), exist_ok=True)#为输入编写对应的输出程序

                data = []
                files = []
                for file in os.listdir(os.path.join(self.data_path, folder)):
                    data.append(json.load(open(os.path.join(self.data_path, folder, file), encoding='utf-8')))
                    files.append(file)#因为文件夹下面还有若干json文件

                data = self.generate_batch(data)
                files = self.generate_batch(files)

                for i, batch in enumerate(data):
                    data[i] = self.add_inputs(batch)
                    for j, d in enumerate(data[i]):
                        with open(os.path.join(self.save_path, folder, files[i][j]), 'w', encoding='utf-8') as f:
                            json.dump(d, f, ensure_ascii=False)

    def generate_batch(self, input_data):
        batches = []
        for idx in range(len(input_data)//self.batch_size+1):
            sub = input_data[idx*self.batch_size:(idx+1)*self.batch_size]
            if sub:
                batches.append(sub)#将数据集中的数据按批输入到batches中

        return batches

    def add_inputs(self, batch):
        facts = []
        for b in batch:
            if 'q' in b:
                facts.append(b['q'])
            else:
                facts.append(b['ajjbqk'])

        # dynamic padding
        # inputs = self.tokenizer.batch_encode_plus(facts, max_length=512)
        # max_length = min(max([len(ipt) for ipt in inputs['input_ids']]), 512)
        max_length = 512
        inputs = self.tokenizer.batch_encode_plus(facts,
                                                  max_length=max_length,#看到这里
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  return_tensors='pt')

        # shift tensors to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        # forward pass
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, pad_token_label_id=-100)#这里的意思是对于padding给他们的id分配
            bio_predictions = outputs[1]
            #如果没有达到-100的话需要在GPU上指定
            #outputs到底是什么，在softmax1之前的得分？
            #这里outputs中的第二项就是得分,详情可见我的jupyter，我不知道什么是pooler
            pad_mask = (bio_predictions != -100).long().to(self.device)
            bio_predictions = bio_predictions * pad_mask#bio_predictions的结果可以在原有基础上做相应的复制

            evt_predictions = (bio_predictions + 1) // 2
#这里用的就是BertCRFForTokenClassification的模型
        inputs['event_type_ids'] = evt_predictions#这里的事件类型应该就是打好的事件类型标签
        for i, b in enumerate(batch):
            input_ids = torch.masked_select(inputs['input_ids'][i], inputs['attention_mask'][i].bool())  # remove <PAD>
            input_ids = input_ids[1:-1]                           # remove the [CLS] and [SEP] tokens
            input_ids = input_ids[0: self.max_length].tolist()    # do truncation to 100 or 409

            event_type_ids = torch.masked_select(inputs['event_type_ids'][i], inputs['attention_mask'][i].bool())
            event_type_ids = event_type_ids[1:-1]
            event_type_ids = event_type_ids[0: self.max_length].tolist()

            if 0 in input_ids:
                print('shit')
            batch[i]['inputs'] = {'input_ids': input_ids,
                                  'event_type_ids': event_type_ids}
        return batch


if __name__ == "__main__":
    random.seed(42)

    os.makedirs('./Downstreams/SCR/SCR-Preprocess/output_data-supervised/query', exist_ok=True)
    os.makedirs('./Downstreams/SCR/SCR-Preprocess/output_data-supervised/candidates', exist_ok=True)

    # process query data
    model = BertCrfTagging(data_path='./Downstreams/SCR/SCR-Preprocess/input_data/data/query/query.json',
                           save_path='./Downstreams/SCR/SCR-Preprocess/output_data-supervised/query/query.json',
                           max_length=100,
                           batch_size=50,
                           mode='query',
                           device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
    model.process()

    # process candidates data
    #candidates data 和query data到底是什么？
    model = BertCrfTagging(data_path='./Downstreams/SCR/SCR-Preprocess/input_data/data/candidates',
                           save_path='./Downstreams/SCR/SCR-Preprocess/output_data-supervised/candidates',
                           max_length=409,
                           batch_size=50,
                           mode='candidate',
                           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.process()
#这是有监督学习，意味着什么？