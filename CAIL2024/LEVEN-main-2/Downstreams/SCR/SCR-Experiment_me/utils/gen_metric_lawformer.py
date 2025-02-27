import os
import json
import math


def ndcg(ranks, K):
    dcg_value = 0.
    idcg_value = 0.
    log_ki = []

    sranks = sorted(ranks, reverse=True)

    for i in range(0, K):
        logi = math.log(i+2, 2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi

    return dcg_value/idcg_value


class Metric:
    def __init__(self, data_path):
        #self.avglist = json.load(open(os.path.join(data_path, "label", "label_top30_dict.json"), "r"))
        self.avglist = json.load(open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的云端硬盘/CAIL2024/LEVEN-main-2/Downstreams/SCR/SCR-Experiment_me/input_data/label/label_order.json', "r"))
        
    
    def NDCG(self, pred, K):
        sndcg = 0.0
        for key in pred.keys():

            key=str(int(float(key)))
            rawranks = [self.avglist[key][str(i)] for i in pred[str(float(int(key)))] if str(i) in self.avglist[key].keys()]
            ranks = rawranks + [0]*(30-len(rawranks))
            if sum(ranks) != 0:
                sndcg += ndcg(ranks, K)
        return round(sndcg/len(pred), 4)
    
    def P(self, pred, K):
        sp = 0.0
        for key in pred.keys():
            key=str(int(float(key)))
            ranks = [i for i in pred[str(float(int(key)))] if str(i) in self.avglist[key].keys()]
            sp += float(len([j for j in ranks[:K] if self.avglist[key][str(j)] == 3])/K)
        return round(sp/len(pred), 4)
    def recall(self,pred,K):
        sp=0.0
        for key in pred.keys():
                key=str(int(float(key)))
                sranks = sorted(self.avglist[key].items(), key=lambda x:x[1],reverse=True)
                #print(sranks)
                sp += float(len([j for j in sranks[:K] if int(j[0]) in pred[str(float(int(key)))][:K]])/K)
        return round(sp/len(pred), 4)
    
    def MAP(self, pred):
        smap = 0.0
        for key in pred.keys():
            key=str(int(float(key)))
            ranks = [i for i in pred[str(float(int(key)))] if str(i) in self.avglist[key].keys()]
            rels = [ranks.index(i) for i in ranks if self.avglist[key][str(i)] == 3]
            tem_map = 0.0
            for rel_rank in rels:
                tem_map += float(len([j for j in ranks[:rel_rank+1] if self.avglist[key][str(j)] == 3])/(rel_rank+1))
            if len(rels) > 0:
                smap += tem_map / len(rels)
        return round(smap/len(pred), 4)

    def pred_path(self, path):
        fnames = os.listdir(path)
        res = {'MAP':0,'Best':-1}
        for fn in fnames:
            fsp = fn.split("-")
            epoch = int(fsp[-1].replace('.json',''))
            #tfile = int(fsp[-2][0])
            metric = {}
            pred = json.load(open(os.path.join(path, fn)))
            for K in [5, 10, 20, 30]:
                metric["NDCG@%d" % K] = self.NDCG(pred, K)
                metric["P%d" % K] = self.P(pred, K)
            metric["MAP"] = self.MAP(pred)
            metric['recall']=self.recall(pred,30)
            
            
            if metric["MAP"] > res["MAP"]:
                res = metric
                
                res["Best"] = epoch
        
        print("==" * 20)
        print(json.dumps(res, ensure_ascii=False, sort_keys=True))

    def pred_single_path(self, path):
        pred = json.load(open(path))

        metric = {}
        for K in [5, 10, 20, 30]:
            metric["NDCG@%d" % K] = self.NDCG(pred, K)
            metric["P%d" % K] = self.P(pred, K)
        metric["MAP"] = self.MAP(pred)
        metric["recall"]=self.recall(pred,30)
        print(json.dumps(metric, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    met = Metric("/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/CAIL2024/LEVEN-main-2/Downstreams/SCR/SCR-Experiment_me/input_data")

    # vanilla BERT
    print('vanilla BERT')
    met.pred_path('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的云端硬盘/CAIL2024/LEVEN-main-2/Downstreams/SCR/SCR-Experiment_me/result/lawformer/test')   # path to the predicted files

    