import json
import torch
import os
import numpy as np

from transformers import AutoTokenizer, BertTokenizer
from formatter.Basic import BasicFormatter


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class PairwiseFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)


    def process(self, data):

        #for pairs in data:
        pairs=[]
        for temp in data:
            for temp_2 in data:
                #Weh just need to get triplets as inputs
                if temp['qid']==temp_2['qid']:
                    if temp['label']>temp_2['label']:
                        pairs.append(temp['feature'],temp_2['feature'],1,temp['index'],temp_2['index'])
                    elif temp['label']<temp_2['label']:
                        pairs.append(temp['feature'],temp_2['feature'],-1,temp['index'],temp_2['index'])
                

        pair_features=np.array([np.concatenate([pair[0], pair[1]]) for pair in pairs])
        pair_label=np.array([pair[2] for pair in pairs])
        pair_index=np.array([np.concatenate([pair[3], pair[4]]) for pair in pairs])

        ret = {
            "inputx": torch.LongTensor(pair_features),
            "labels": torch.LongTensor(pair_label),
            "index":torch.longTensor(pair_index)
        }
      
        return ret
