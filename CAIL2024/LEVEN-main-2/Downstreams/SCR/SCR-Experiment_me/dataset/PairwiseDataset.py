import json
import os
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
import random


class PairwiseDataset(IterableDataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.query_path = config.get("data", "query_path")
        self.cand_path = config.get("data", "cand_path")
        self.labels = json.load(open(config.get("data", "label_path"),'r'))
        self.data = []

        self.test_file = config.get("data", "test_file")
        querys = []
        if mode=='train':
            with open(self.query_path,'r') as f:
                for line in f:
                    querys.append(json.loads(line))
        else:
            with open(self.test_file,'r') as f:
                for line in f:
                    querys.append(json.loads(line))
        pos_num = 0
        self.query2posneg = {}
        print(len(querys))
        for query in querys:
            if str(query['id']) not in self.labels.keys():
                continue
            que = query["fact_part"]
            path = os.path.join(self.cand_path,str(query['id']))
            #path = os.path.join(self.cand_path)
            self.query2posneg[str(query["id"])] = {"pos": [], "neg": []}
            for fn in os.listdir(path):
                try:
                    cand = json.load(open(os.path.join(path, fn), "r"))
                except json.decoder.JSONDecodeError:
                    continue
                label=str(int(fn.split('.')[0])) in self.labels[str(query["id"])]
                
                self.data.append({
                    "query": que,
                    "cand": cand["fact_part"],
                    "label": label,
                    "index": (query["id"], fn.split('.')[0]),
                    "query_inputs": query['inputs'],                # added event info
                    "cand_inputs": cand['inputs']                   # added event info
                })
                pos_num += int(label)
        print(mode, "positive num:", pos_num)

   # def __getitem__(self, item):
    #    pair1 = self.data[item % len(self.data)]
     #   return (pair1, )
    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        if self.mode == "train":
            return len(self.data)
        else:
            return len(self.data)