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
        self.avglist = json.load(open("/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的云端硬盘/LEVEN-main/Downstreams/SCR/SCR-Experiment/input_data/label/label_top30_dict.json", "r"))
        self.combdic = json.load(open("/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的云端硬盘/LEVEN-main/Downstreams/SCR/SCR-Experiment/input_data/prediction/combined_top100.json", "r"))
    
    
    def NDCG(self, pred, K):
        #print(self.combdic.keys())
        sndcg = 0.0
        for key in pred.keys():
            rawranks = [self.avglist[key][str(i)] for i in pred[key] if i in list(self.combdic[key][:30])]
            ranks = rawranks + [0]*(30-len(rawranks))
            if sum(ranks) != 0:
                sndcg += ndcg(ranks, K)
        return round(sndcg/len(pred), 4)
    
    def P(self, pred, K):
        sp = 0.0
        for key in pred.keys():
      
            ranks = [i for i in pred[key] if i in list(self.combdic[key][:30])]
            sp += float(len([j for j in ranks[:K] if self.avglist[key][str(j)] == 3])/K)
        return round(sp/len(pred), 4)
    
    def MAP(self, pred):
        smap = 0.0
        for key in pred.keys():
            ranks = [i for i in pred[key] if i in list(self.combdic[key][:30])]
            rels = [ranks.index(i) for i in ranks if self.avglist[key][str(i)] == 3]
            tem_map = 0.0
            for rel_rank in rels:
                tem_map += float(len([j for j in ranks[:rel_rank+1] if self.avglist[key][str(j)] == 3])/(rel_rank+1))
            if len(rels) > 0:
                smap += tem_map / len(rels)
        return round(smap/len(pred), 4)


    def pred_path(self, path):#这个函数和下面那个函数有什么区别
        fnames = os.listdir(path)
        res = {}
        for fn in fnames:
            if '.json' in fn and 'Lawformer' in fn:
                test_file=fn.split("-")[2].replace('_epoch','')
                epoch=float(fn.split('-')[3].replace('.json',''))

            else:
                continue
            
            metric = {}
            pred = json.load(open(os.path.join(path, fn)))
            for K in [5, 10, 20, 30]:
                metric["NDCG@%d" % K] = self.NDCG(pred, K)
                metric["P%d" % K] = self.P(pred, K)
            #metric['recall']=self.recall(pred,30)
            metric["MAP"] = self.MAP(pred)
           
            if test_file not in res:
               res[test_file] = {"MAP": -1, "best": -1}
            if metric["MAP"] > res[test_file]["MAP"]:
               res[test_file] = metric
               res[test_file]['best']=epoch
        print(res)
        overall={}
        for round in res:
            for key in res[round]:
                    if key not in overall:
                            overall[key] = 0
                    overall[key]+= res[round][key]
        for key in overall:
            overall[key] /= 5
       
        print("==" * 20)
        print(json.dumps(overall, ensure_ascii=False, sort_keys=True))

    def recall(self,pred,K):
            sp=0.0
            for key in pred.keys():
                    sranks = sorted(self.avglist[key].items(), key=lambda x:x[1],reverse=True)
                    #print(sranks)
                    sp += float(len([j for j in sranks[:K] if int(j[0]) in pred[key][:K]])/K)
            return round(sp/len(pred), 4)
    def pred_single_path(self, path):
        pred = json.load(open(path))

        metric = {}
        for K in [5, 10, 20, 30]:
            metric["NDCG@%d" % K] = self.NDCG(pred, K)
            metric["P%d" % K] = self.P(pred, K)
        metric["MAP"] = self.MAP(pred)
        metric['recall']=self.recall(pred,30)
        print(json.dumps(metric, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    met = Metric("./input_data")
    
    met.pred_path('/Users/yuqi/Downloads/lawformer_result/test')
    