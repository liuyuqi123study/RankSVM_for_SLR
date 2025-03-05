import os
import sys
import copy
import json
import torch
from torch.utils.data import IterableDataset

from utils import flatten_concatenation,flatten_concatenation_batch

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, tokenizer, args):
        self.data = input_data
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return {
            "input_text": self.data[index][0],
            "label": self.data[index][1],
        }
    
    def collate_fn(self, batch):
        # enable smart batching
        batch_text = [row["input_text"] for row in batch]
        batch_label = [row["label"] for row in batch]
        tokenized_input = self.tokenizer(
                                batch_text, 
                                padding=True, 
                                truncation="longest_first",
                                max_length=args.max_length, 
                                return_tensors="pt",
                                )
        return {
            "source": tokenized_input,
            "target": torch.tensor(batch_label),
        }

class LCEDatasetMaskedLM(torch.utils.data.Dataset):
    def __init__(self, collection, queries, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.collection = collection
        self.queries = queries
        self.sep_token = tokenizer.sep_token if tokenizer.sep_token else tokenizer.eos_token
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        line = self.dataset[index]
        input_pretokenized = []
        for pid in line[1:]:
            input_pretokenized.append(self.queries[line[0]]+self.sep_token+self.collection[pid])
        return input_pretokenized
    
    def collate_fn(self, batch):
        input_pretokenized = flatten_concatenation(batch)
        tokenized_input = self.tokenizer(input_pretokenized, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return tokenized_input

class LCEDatasetCausalLM(IterableDataset):
    def __init__(self, args,tokenizer, max_length=1024,collection=None,queries=None,dataset=None,mode='train'):
        self.dataset=dataset
        #传入的时候就应该已经存储为列表了
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.collection = collection
        self.sep_token = "\n\n"
        self.labels=json.load(open('/content/drive/MyDrive/LEVEN-main/Downstreams/SCR/SCR-Experiment/input_data/label/golden_labels.json','r'))
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"
        self.cand_path=os.path.join(args.input_dir,'candidates')
        self.data=[]
        self.mode=mode


        querys=[]
        for i in range(5):
            if mode=='train':
                if i!=args.fold:
                    with open(os.path.join(args.input_dir,'query','query_5fold',"query_"+str(i)+'.json'), "r") as fin:
                            querys+=json.load(fin)
            else:
                if i== args.fold:
                    with open(os.path.join(args.input_dir,'query','query_5fold',"query_"+str(i)+'.json'), "r") as fin:
                            querys+=json.load(fin)
        pos_num = 0
        self.query2posneg = {}
        for query in querys:
            que=query["q"]
            path = os.path.join(self.cand_path, str(query["ridx"]))
            self.query2posneg[str(query["ridx"])] = {"pos": [], "neg": []}
            for fn in os.listdir(path):
                    cand=json.load(open(os.path.join(path, fn), "r"))
                    label = int(fn.split('.')[0]) in self.labels[str(query["ridx"])]
                    if label:
                        self.query2posneg[str(query["ridx"])]["pos"].append(len(self.data))
                    else:
                        self.query2posneg[str(query["ridx"])]["neg"].append(len(self.data))
            
            
                    document = self.tokenizer(cand['ajjbqk'], truncation=True, max_length=args.cand_length)  # hardcoded
                    truncated_document = self.tokenizer.decode(document.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    query_token=self.tokenizer(que, truncation=True, max_length=args.query_length)  # hardcoded
                    truncated_query=self.tokenizer.decode(query_token.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    input_pretokenized=truncated_document+self.sep_token+truncated_query+self.tokenizer.eos_token
                    self.data.append({
                        'inputx':input_pretokenized,
                        'label':label,
                        'index':(query['ridx'],fn.split('.')[0])
                    })
                    pos_num+=int(label)
        print(len(self.data))
        print(mode,'positive num:',pos_num)
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def collate_fn(self, batch):
        inputx=[]
        labels=[]
        for temp in batch:
           
            labels.append(temp['label'])
            inputx.append(temp['inputx'])
        tokenized_input = self.tokenizer(inputx, padding=True, truncation=False, return_tensors="pt")
        ret={
            'inputx':tokenized_input,
            'labels':torch.LongTensor(labels),
            
        }
        ret["index"] = [temp["index"] for temp in batch]
        return ret

class LCEDatasetSeq2SeqLM(torch.utils.data.Dataset):
    def __init__(self, collection, queries, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.collection = collection
        self.queries = queries
        self.sep_token = tokenizer.sep_token if tokenizer.sep_token else tokenizer.eos_token
        self.bos_token, self.bos_token_id = tokenizer.pad_token, tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        line = self.dataset[index]
        input_pretokenized = []
        for pid in line[1:]:
            input_pretokenized.append(self.bos_token+self.queries[line[0]]+self.sep_token+self.collection[pid])
        return input_pretokenized
    
    def collate_fn(self, batch):
        input_pretokenized = flatten_concatenation(batch)
        tokenized_input = self.tokenizer(input_pretokenized, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return tokenized_input