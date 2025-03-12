import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
from sklearn.metrics import roc_curve,auc
#We need to make sure they are in the same order, which is not really true especially for my RankSVM inputs
# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker
import json

score_train=[]
#first step:getting the prediction score from the training set
with open ('RankMamba/Visualization/predictions_0_1_train.dat') as f1:
    for line in f1:
        score_train.append(float(line.strip()))

#首先要对csv文件进行读取
score_svm_train=pd.DataFrame()
j=0

#second step：getting the score from the training set and mamba model
score_mamba_train = pd.read_csv('RankMamba/Visualization/train_score_mamba0.csv')
#2.5 step: remove some redundant headers.
score_mamba_train=score_mamba_train[score_mamba_train['qid']!='qid']
score_mamba_train.to_csv('RankMamba/Visualization/train_score_mamba0.csv',index=False)
score_mamba_train['qid']=score_mamba_train['qid'].astype(int)
#2.75 step: need to remove some strange things around the label
score_mamba_train['label']=score_mamba_train['label'].str.replace('tensor','').str.strip('(').str.strip(')').astype(int)

score_mamba_train=score_mamba_train.sort_values(by='qid')
#We need to load
#third step: combine the ind and label of rank svm and the score
with open('RankMamba/features/train_feature_modified0.dat') as f:
    for line in f:
                    score_svm_train.loc[j,'cid']=int(line.split('#')[1].replace('\n','').strip('"'))
                    score_svm_train.loc[j,'qid']=int(line[line.find('qid')+4:line.find(':',line.find('qid')+4)-2])-5181
                    score_svm_train.loc[j,'score']=score_train[j]
                    score_svm_train.loc[j,'label']=int(line.split()[0])
                    j=j+1
#我们首先从我们的特征中确定我们的需要的index

#step 3.5:
#we need to reorder the list so that it is the same as RankSVM
#fourth step:
#read the ranksvm result for the test dataset

S_color_2_test=[]

with open ('RankMamba/Visualization/predictions_0_1_test.dat') as f2:
    for line in f2:
        S_color_2_test.append(float(line.strip()))
#对这种方法下的得分进行标识
#fifth step: get the result of the mamba on the test dataset

score_mamba_test = pd.read_csv('RankMamba/Visualization/test_score_mamba0.csv')
score_mamba_test=score_mamba_test[score_mamba_test['qid']!='qid']
score_mamba_test.to_csv('RankMamba/Visualization/test_score_mamba0.csv',index=False)
score_mamba_test['qid']=score_mamba_test['qid'].astype(int)
score_mamba_test['label']=score_mamba_test['label'].str.replace('tensor','').str.strip('(').str.strip(')').astype(int)

score_mamba_test=score_mamba_test.sort_values(by='qid')
score_svm_test=pd.DataFrame()
j=0
with open('RankMamba/features/test_feature_modified0.dat') as f:
    for line in f:
                    score_svm_test.loc[j,'cid']=int(line.split('#')[1].replace('\n','').strip('"'))
                    score_svm_test.loc[j,'qid']=int(line[line.find('qid')+4:line.find(':',line.find('qid')+4)-2])-5181
                    score_svm_test.loc[j,'score']=S_color_2_test[j]
                    score_svm_test.loc[j,'label']=int(line.split()[0])
                    j=j+1


#然后我们进行绘制
#getting the score
fpr_train_mamba,tpr_train_mamba,thresholds=roc_curve(score_mamba_train['label'].tolist(),score_mamba_train['score'].tolist())
roc_auc_train_mamba=auc(fpr_train_mamba,tpr_train_mamba)

fpr_train_SVM,tpr_train_SVM,thresholds=roc_curve(score_svm_train['label'],score_svm_train['score'])
roc_auc_train_SVM=auc(fpr_train_SVM,tpr_train_SVM)
plt.figure(figsize=(8, 6))
plt.plot(fpr_train_mamba, tpr_train_mamba, color='blue', lw=2, label=f'ROC Curve for mamba(AUC = {roc_auc_train_mamba:.2f})')
plt.plot(fpr_train_SVM,tpr_train_SVM,color='yellow', lw=2, label=f'ROC Curve for RankSVM(AUC = {roc_auc_train_SVM:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random performance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for training set(mamba)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('mamba_train_roc.pdf')
plt.show()

#然后我们进行绘制
fpr_test_mamba,tpr_test_mamba,thresholds=roc_curve(score_mamba_test['label'].tolist(),score_mamba_test['score'].tolist())
roc_auc_test_mamba=auc(fpr_test_mamba,tpr_test_mamba)

fpr_test_SVM,tpr_test_SVM,thresholds=roc_curve(score_svm_test['label'],score_svm_test['score'])
roc_auc_test_SVM=auc(fpr_test_SVM,tpr_test_SVM)
plt.figure(figsize=(8, 6))
plt.plot(fpr_test_mamba, tpr_test_mamba, color='blue', lw=2, label=f'ROC Curve for mamba(AUC = {roc_auc_test_mamba:.2f})')
plt.plot(fpr_test_SVM,tpr_test_SVM,color='yellow', lw=2, label=f'ROC Curve for RankSVM(AUC = {roc_auc_test_SVM:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random performance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for test set(mamba)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('mamba_test_roc.pdf')
plt.show()

#Getting ROC for Mamba










                              


