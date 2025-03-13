import json
import os
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
import random


#class PairwiseDataset(Dataset):
class TripletDataset(IterableDataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.feature_path=config.get("data","feature_path")
        self.data = []

        test_file = config.get("data", "test_file")
        querys = []
        for i in range(5):
            if mode == 'train':
                if test_file == str(i):
                    continue
                else:
                    querys += json.load(open(os.path.join(self.query_path, 'query_%d.json' % i), 'r'))
            else:
                if test_file == str(i):
                    querys += json.load(open(os.path.join(self.query_path, 'query_%d.json' % i), 'r'))
        pos_num = 0
        self.query2posneg = {}
        for query in querys:
            que = query["q"]
            path = os.path.join(self.cand_path, str(query["ridx"]))#去这个文件夹下面找,在candidates中存放的是100个候选文件
            self.query2posneg[str(query["ridx"])] = {"pos": [], "neg": []}
            for fn in os.listdir(path):
                cand = json.load(open(os.path.join(path, fn), "r"))
                label = int(fn.split('.')[0]) in self.labels[str(query["ridx"])]
                if label:
                    self.query2posneg[str(query["ridx"])]["pos"].append(len(self.data))
                else:
                    self.query2posneg[str(query["ridx"])]["neg"].append(len(self.data))
                if 'inputs' in query.keys():
                    self.data.append({
                        "query": que,
                        "cand": cand["ajjbqk"],
                        "label": label,
                        "index": (query["ridx"], fn.split('.')[0]),
                        "query_inputs": query['inputs'],                # added event info
                        "cand_inputs": cand['inputs']                   # added event info
                    })
                else:
                    self.data.append({
                        "query": que,
                        "cand": cand["ajjbqk"],
                        "label": label,
                        "index": (query["ridx"], fn.split('.')[0]),
                           
                    })
                pos_num += int(label)
        print(mode, "positive num:", pos_num)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        if self.mode == "train":
            return len(self.data)
        else:
            return len(self.data)
