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
        #self.avglist = json.load(open(os.path.join(data_path, "label", "label_order.json"), "r"))
        self.avglist=json.load(open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的云端硬盘/CAIL2024/LEVEN-main-2/Downstreams/SCR/SCR-Experiment_me/input_data/label/label_order.json'))
        #self.combdic = json.load(open("/Users/yuqi/Downloads/LEVEN-main/Downstreams/SCR/SCR-Experiment/input_data/prediction/combined_top100.json", "r"))
    
    def NDCG(self, pred, K):
        #print(self.combdic.keys())
        sndcg = 0.0
        for key in pred.keys():
            key=str(int(float(key)))
            #print('-5180' in self.combdic.keys())
            #print(self.avglist[key])
            rawranks = [self.avglist[key][str(i)] for i in pred[str(float(int(key)))] if str(i) in list(self.avglist[key].keys())]
            ranks = rawranks + [0]*(30-len(rawranks))
            if sum(ranks) != 0:
                sndcg += ndcg(ranks, K)
        return round(sndcg/len(pred), 4)
    
    def P(self, pred, K):
        sp = 0.0
        for key in pred.keys():
            key=str(int(float(key)))
            #ranks = [i for i in pred[] if str(i) in list(self.avglist[key].keys())]
            #sp += float(len([j for j in ranks[:K] if self.avglist[key][str(j)] == 3])/K)
            ranks = [i for i in pred[str(float(int(key)))]]
            sp += float(len([j for j in ranks[:K] if str(j) in self.avglist[key].keys()])/K)
        return round(sp/len(pred), 4)
    
    def MAP(self, pred):
        smap = 0.0
        for key in pred.keys():
            key=str(int(float(key)))
            ranks = [i for i in pred[str(float(int(key)))] if str(i) in list(self.avglist[key].keys())]
            rels = [ranks.index(i) for i in ranks if self.avglist[key][str(i)] == 3]
            tem_map = 0.0
            for rel_rank in rels:
                tem_map += float(len([j for j in ranks[:rel_rank+1] if self.avglist[key][str(j)] == 3])/(rel_rank+1))
            if len(rels) > 0:
                smap += tem_map / len(rels)
        return round(smap/len(pred), 4)
   # def accuracy(self,pred):
    #    smap = 0.0
     #   for key in pred.keys():
      #      key=str(int(float(key)))
       #     ranks = [i for i in pred[str(float(int(key)))] if i in list(self.combdic[key][:30])]
        #    smap+=1-(30-len(ranks))*2/100
        #return round(smap/len(pred), 4)

    def pred_path(self, path):#这个函数和下面那个函数有什么区别
        fnames = os.listdir(path)
        res = {}
        for fn in fnames:
            if '.json' in fn and 'article' in fn:
                if 'w' in fn:
                    w=fn.split("w")[1][0]
                #print(fsp[1])
                #model_name==int(fsp[2].replace('test',''))
                    C=fn.split('w')[0].replace('rank_v2_130_','')
                else:
                    w=str(3)
                    C=fn.split('_')[2].replace('detail','')

            else:
                continue
            #if C==0:
             #   continue
            #epoch = int(fsp[-1][0])
            #tfile = int(fsp[-2][0])
            metric = {}
            pred = json.load(open(os.path.join(path, fn)))
            for K in [5, 10, 20, 30]:
                metric["NDCG@%d" % K] = self.NDCG(pred, K)
                metric["P%d" % K] = self.P(pred, K)
            metric['recall']=self.recall(pred,30)
            metric["MAP"] = self.MAP(pred)
            #metric['accuracy']=self.accuracy(pred)
            #modelname = fsp[0]
            #if C not in res:
             #  res[C] = {}
            #if C not in res:
             #   res[C]={}
            #if C not in res:
            if w not in res:
               res[w] = {"MAP": -1, "best": -1,'recall':0}
            #res = {"MAP": -1,'best':0}
            #print(C)
            #print(metric)
            if metric["MAP"] > res[w]["MAP"]:
               res[w] = metric
               res[w]['best']=C
              #  res[modelname][tfile]["best"] = epoch
        #for model in res:
        print(res)
        '''overall={}
        for round in res:
            #for tf in res[model]:
            for key in res[round]:
                    if key not in overall:
                            overall[key] = 0
                    overall[key]+= res[round][key]
        
        for key in overall:
            overall[key] /= 5
            #print(C)'''
        print("==" * 20)
        print(json.dumps(res, ensure_ascii=False, sort_keys=True))

    def recall(self,pred,K):
            sp=0.0
            for key in pred.keys():
                    key=str(int(float(key)))
                    sranks = sorted(self.avglist[key].items(), key=lambda x:x[1],reverse=True)
                    #print(sranks)
                    sp += float(len([j for j in sranks[:K] if int(j[0]) in pred[str(float(int(key)))][:K]])/K)
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
    met = Metric("/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/CAIL2024/LEVEN-main-2/Downstreams/SCR/SCR-Experiment_me/input_data")

    # vanilla BERT
    #print('vanilla BERT')
    #met.pred_path('/Users/yuqi/Downloads/LEVEN-main/Downstreams/SCR/SCR-Experiment/result/BERT/test0')   # path to the predicted files
    #met.pred_path('/Users/yuqi/Downloads/svm_rank/LeCaRDv1')
    # BERT with event
    #print('BERT with event')
    met.pred_path('/Users/yuqi/Downloads/svm_rank/LeCaRDv2')
    #met.pred_single_path('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的雲端硬碟/CAIL2024/LEVEN-main-2/result/EDBERT/test_whole/prediction_200.json')
    #for i in range(5):
     #   for m in [0.02,0.05,0.1,0.5,1]:
      ##     print(path)
        #    met.pred_single_path(path)