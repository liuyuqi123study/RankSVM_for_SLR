import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, manifold

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker

#在这里对第一折的特征进行读取
csvfile_1=pd.read_csv("/Users/yuqi/Downloads/feature_test/features_v1_train_BERT0.csv").iloc[:,2:-1]
csvfile_2=pd.read_csv("/Users/yuqi/Downloads/feature_test/features_v1_test_BERT0.csv").iloc[:,2:-1]
X_embedded_1=TSNE(n_components=2).fit_transform(csvfile_1)

def add_2d_scatter(ax, points, points_color,label=None, title=None,cmap=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=5, alpha=1,label=label,cmap=cmap)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
def plot_2d(points, points_color, title,label=None,cmap=None):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color,label=label,cmap=cmap)
    plt.show()
def add_2d_scatter_label(ax, x_1,y_1, x_0,y_0,label=None, title=None,cmap=None):
    ax.scatter(x_1, y_1, c='#1f77b4', s=5, alpha=1,label=label[1])
    ax.scatter(x_0,y_0,c='#ff7f0e', s=5, alpha=1,label=label[0])
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
def plot_2d_label(x_1,y_1,x_0,y_0,title,label=None,cmap=None):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter_label(ax, x_1,y_1,x_0,y_0,label=label,cmap=cmap)
    plt.show()
#n_samples=8512
S_color = pd.read_csv("/Users/yuqi/Downloads/feature_test/features_v1_train_BERT0.csv").iloc[:,0]
label_train=['irrelevant','relevant']
train_dataset=pd.DataFrame({'x':X_embedded_1[:,0],'y':X_embedded_1[:,1],'label':S_color})
train_dataset_relevant=train_dataset[train_dataset['label']==1]
train_dataset_irrelevant=train_dataset[train_dataset['label']==0]
plot_2d_label(train_dataset_relevant['x'],train_dataset_relevant['y'],train_dataset_irrelevant['x'],train_dataset_irrelevant['y'], "T-distributed Stochastic  \n Neighbor Embedding for labels",label_train)
S_color_2=[]
#prediction for svm
with open ('/Users/yuqi/Downloads/svm_rank/prediction_v1_BERT_0_train_0.02.txt') as f1:
    for line in f1:
        S_color_2.append(float(line.strip()))
plot_2d(X_embedded_1, S_color_2, "T-distributed Stochastic  \n Neighbor Embedding for RankSVM+BERT",cmap='Greens')
#首先要对csv文件进行读取
score=pd.DataFrame()
j=0
data_1 = pd.read_csv('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的云端硬盘/LEVEN-main/Downstreams/SCR/SCR-Experiment/score_train.csv')
with open('/Users/yuqi/Downloads/feature_test/features_v1_train_BERT_0.dat') as f:
    for line in f:
                    score.loc[j,'id']=int(line.split('#')[1])
                    score.loc[j,'qid']=int(line[line.find('qid')+4:line.find(':',line.find('qid')+4)-2])-5181
                    for k in range(len(data_1)):
                        if score.loc[j,'id']==int(data_1.loc[k,'index'][1:-1].split(',')[1].strip().strip("'")) and score.loc[j,'qid']==int(data_1.loc[k,'index'][1:-1].split(',')[0]):
                            
                              score.loc[j,'score']=float(data_1.loc[k,'score'])#把他们存到一个表里面
                              break
                        
                    j=j+1
S_color_3=score.loc[:,'score']#这个顺序应该和特征的顺序是一样的
plot_2d(X_embedded_1, S_color_3, "T-distributed Stochastic  \n Neighbor Embedding for BERT",cmap='Greens')

#接下来画测试数据
X_embedded_2=TSNE(n_components=2).fit_transform(csvfile_2)
#重新获得进行维度转换后的数据
S_color_1_test=pd.read_csv("/Users/yuqi/Downloads/feature_test/features_v1_test_BERT0.csv").iloc[:,0]
label_test=['irrelevant','relevant']
test_dataset=pd.DataFrame({'x':X_embedded_2[:,0],'y':X_embedded_2[:,1],'label':S_color_1_test})
test_dataset_relevant=test_dataset[test_dataset['label']==1]
test_dataset_irrelevant=test_dataset[test_dataset['label']==0]
plot_2d_label(test_dataset_relevant['x'],test_dataset_relevant['y'],test_dataset_irrelevant['x'],test_dataset_irrelevant['y'], "T-distributed Stochastic  \n Neighbor Embedding for labels test",label_test)
#得到ranksvm的得分
S_color_2_test=[]
with open ('/Users/yuqi/Downloads/svm_rank/prediction_v1_BERT_0_0.02.txt') as f2:
    for line in f2:
        S_color_2_test.append(float(line.strip()))
#对这种方法下的得分进行标识
plot_2d(X_embedded_2, S_color_2_test, "T-distributed Stochastic  \n Neighbor Embedding for RankSVM+BERT test",cmap='Greens')
score_test=pd.DataFrame()
j=0
data_2 = pd.read_csv('/Users/yuqi/Library/CloudStorage/GoogleDrive-yuqi5341@gmail.com/我的云端硬盘/LEVEN-main/Downstreams/SCR/SCR-Experiment/score_test.csv')
#对id和qid进行读取
with open('/Users/yuqi/Downloads/feature_test/features_v1_test_BERT_0.dat') as f:
    for line in f:
                    score_test.loc[j,'id']=int(line.split('#')[1])
                    score_test.loc[j,'qid']=int(line[line.find('qid')+4:line.find(':',line.find('qid')+4)-2])-5181
                    
                    for k in range(len(data_2)):
                        #print(data_1.loc[k,'index'][0])
                        if score_test.loc[j,'id']==int(data_2.loc[k,'index'][1:-1].split(',')[1].strip().strip("'")) and score_test.loc[j,'qid']==int(data_2.loc[k,'index'][1:-1].split(',')[0]):
                              score_test.loc[j,'score']=float(data_2.loc[k,'score'])#把他们存到一个表里面
                              break
                        k=k+1
                    j=j+1
S_color_3_test=score_test.loc[:,'score']#这个顺序应该和特征的顺序是一样的
plot_2d(X_embedded_2, S_color_3_test, "T-distributed Stochastic  \n Neighbor Embedding for BERT test",cmap='Greens')







                              
