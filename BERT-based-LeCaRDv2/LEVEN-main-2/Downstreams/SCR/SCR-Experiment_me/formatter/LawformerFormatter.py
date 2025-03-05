import json
import torch
import os
import numpy as np

import random
from transformers import AutoTokenizer

class LawformerFormatter:
    def __init__(self, config, mode, *args, **params):
        
        self.query_len=config.getint('train','query_len')
        self.cand_len=config.getint('train','cand_len')
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_vocab'))
        #这里还需要我们对输入数据的格式进行处理
        
        self.max_len=self.query_len+self.cand_len+3
        self.pad_id=self.tokenizer.pad_token_id
        self.sep_id=self.tokenizer.sep_token_id
        self.cls_id=self.tokenizer.cls_token_id

    def process(self, data, config, mode, *args, **params):
       
        inputx = []
        segment=[]
        mask=[]
        labels=[]
        
        for temp in data:
            inputx.append([])
            segment.append([])
            mask.append([])
            query = self.tokenizer.tokenize(temp["query"])[:self.query_len]
            cand = self.tokenizer.tokenize(temp["cand"])[:self.cand_len]
            tokens = ["[CLS]"] + query + ["[SEP]"] + cand + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * (len(query) + 2) + [1] * (len(cand) + 1)
            input_mask = [1] * len(input_ids)
        
            padding = [0] * (self.max_len - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding


            assert len(input_ids) == self.max_len
            assert len(input_mask) == self.max_len
            assert len(segment_ids) == self.max_len
            
            inputx[-1].append(input_ids)
            segment[-1].append(segment_ids)
            mask[-1].append(input_mask)
            labels.append(temp['label'])
        if mode == "train":
            global_att = np.zeros((len(data), 2, self.max_len), dtype=np.int32)
        else:
            global_att = np.zeros((len(data), 1, self.max_len), dtype=np.int32)
        global_att[:, :, 0] = 1

        ret = {
            "inputx": torch.LongTensor(inputx),
            "segment": torch.LongTensor(segment),
            "mask": torch.LongTensor(mask),
            #"event": torch.LongTensor(event) if use_event else None,
            "global_att": torch.LongTensor(global_att),
            "labels": torch.LongTensor(labels),
        }
        ret["index"] = [temp["index"] for temp in data]
        return ret