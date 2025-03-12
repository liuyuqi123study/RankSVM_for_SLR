import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
from sklearn.metrics import roc_curve,auc
# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker
import json

score_train=[]
with open ('/Users/yuqi/Downloads/svm_rank/prediction_v1_lawformer_0.001_train.txt') as f1:
    for line in f1:
        score_train.append(float(line.strip()))
#得到预测的得分
#首先要对csv文件进行读取
score_svm_train=pd.DataFrame()
j=0
data_1 = pd.read_csv('/Users/yuqi/Downloads/score_train_lawformer_v1_fold_4.csv')
with open('/Users/yuqi/Downloads/feature_test/features_v1_train_lawformer_4.dat') as f:
    for line in f:
                    score_svm_train.loc[j,'id']=int(line.split('#')[1])
                    score_svm_train.loc[j,'qid']=int(line[line.find('qid')+4:line.find(':',line.find('qid')+4)-2])-5181
                    score_svm_train.loc[j,'score']=score_train[j]
                    score_svm_train.loc[j,'label']=int(line.split()[0])
                    j=j+1
#我们首先从我们的特征中确定我们的需要的index

score_Lawformer_train=pd.DataFrame()

#我们再在这里直接对lawformer的score进行读取
labels=json.load(open('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的云端硬盘/LEVEN-main/Downstreams/SCR/SCR-Experiment/input_data/label/golden_labels.json','r'))
for k in range(len(data_1)):
       score_Lawformer_train.loc[k,'score']=data_1.loc[k,'score']
       cid=int(data_1.loc[k,'index'][1:-1].split(',')[1].strip().strip("'"))
       qid=data_1.loc[k,'index'][1:-1].split(',')[0]
       label=(cid in labels[qid])
       score_Lawformer_train.loc[k,'label']=label



S_color_2_test=[]
with open ('/Users/yuqi/Downloads/svm_rank/prediction_v1_lawformer_0.001_4.txt') as f2:
    for line in f2:
        S_color_2_test.append(float(line.strip()))
#对这种方法下的得分进行标识

score_Lawformer_test=pd.DataFrame()
data_2 = pd.read_csv('/Users/yuqi/Downloads/score_test_lawformer_v1_fold_4.csv')
#对id和qid进行读取
for k in range(len(data_2)):
       score_Lawformer_test.loc[k,'score']=data_2.loc[k,'score']
       cid=int(data_2.loc[k,'index'][1:-1].split(',')[1].strip().strip("'"))
       qid=data_2.loc[k,'index'][1:-1].split(',')[0]
       label=(cid in labels[qid])
       score_Lawformer_test.loc[k,'label']=label
j=0
score_svm_test=pd.DataFrame()

with open('/Users/yuqi/Downloads/feature_test/features_v1_test_lawformer_4.dat') as f:
    for line in f:
                    score_svm_test.loc[j,'id']=int(line.split('#')[1])
                    score_svm_test.loc[j,'qid']=int(line[line.find('qid')+4:line.find(':',line.find('qid')+4)-2])-5181
                    score_svm_test.loc[j,'score']=S_color_2_test[j]
                    score_svm_test.loc[j,'label']=int(line.split()[0])
                    j=j+1


#然后我们进行绘制
fpr_train_BERT,tpr_train_BERT,thresholds=roc_curve(score_Lawformer_train['label'].tolist(),score_Lawformer_train['score'].tolist())
roc_auc_train_BERT=auc(fpr_train_BERT,tpr_train_BERT)

fpr_train_SVM,tpr_train_SVM,thresholds=roc_curve(score_svm_train['label'],score_svm_train['score'])
roc_auc_train_SVM=auc(fpr_train_SVM,tpr_train_SVM)
plt.figure(figsize=(8, 6))
plt.plot(fpr_train_BERT, tpr_train_BERT, color='blue', lw=2, label=f'ROC Curve for Lawformer(AUC = {roc_auc_train_BERT:.2f})')
plt.plot(fpr_train_SVM,tpr_train_SVM,color='yellow', lw=2, label=f'ROC Curve for SVM(AUC = {roc_auc_train_SVM:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random performance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for training set')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

#然后我们进行绘制
fpr_test_BERT,tpr_test_BERT,thresholds=roc_curve(score_Lawformer_test['label'].tolist(),score_Lawformer_test['score'].tolist())
roc_auc_test_BERT=auc(fpr_test_BERT,tpr_test_BERT)

fpr_test_SVM,tpr_test_SVM,thresholds=roc_curve(score_svm_test['label'],score_svm_test['score'])
roc_auc_test_SVM=auc(fpr_test_SVM,tpr_test_SVM)
plt.figure(figsize=(8, 6))
plt.plot(fpr_test_BERT, tpr_test_BERT, color='blue', lw=2, label=f'ROC Curve for Lawformer(AUC = {roc_auc_test_BERT:.2f})')
plt.plot(fpr_test_SVM,tpr_test_SVM,color='yellow', lw=2, label=f'ROC Curve for SVM(AUC = {roc_auc_test_SVM:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random performance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for test set')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

#Getting ROC for Mamba










                              


