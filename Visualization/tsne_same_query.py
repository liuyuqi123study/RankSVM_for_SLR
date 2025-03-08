import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, manifold

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker

#在这里对第一折的特征进行读取
#csvfile_1=pd.read_csv("/Users/yuqi/Downloads/feature_test/features_v1_train_BERT0.csv").iloc[:,2:-1]
csvfile_2=pd.read_csv("/Users/yuqi/Downloads/feature_test/features_v1_test_BERT0.csv").iloc[:,2:-1]
#X_embedded_1=TSNE(n_components=2).fit_transform(csvfile_1)

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
with open ('/Users/yuqi/Downloads/svm_rank/prediction_v1_BERT_0.1_0_same_query.txt') as f2:
    for line in f2:
        S_color_2_test.append(float(line.strip()))
#对这种方法下的得分进行标识
plot_2d(X_embedded_2, S_color_2_test, "T-distributed Stochastic  \n Neighbor Embedding for RankSVM+BERT test",cmap='Greens')








                              
