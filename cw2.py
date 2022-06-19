import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
from sklearn.preprocessing import minmax_scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv('/Users/ljcmhw/Desktop/INT104/cw2/CW_Data.csv').dropna()
# df.drop(labels="ID", axis=1, inplace=True)
# df.drop(df[df['Programme'].isnull()].index, inplace=True)
data=df.values
# print(data)
Class_0=[]
Class_1=[]
Class_2=[]
Class_3=[]
Class_4=[]

# PCA （减少数据版对应的输入数据）
# for i in range(len(data)):
#     if data[i,-1] == 0:
#         Class_0.append(data[i,[1,2]])
#     elif data[i,-1] == 1:
#         Class_1.append(data[i,[1,4]])
#     elif data[i,-1] == 2:
#         Class_2.append(data[i,[2,5]])
#     elif data[i,-1] == 3:
#         Class_3.append(data[i,[1,4]])
#     elif data[i,-1] == 4:
#         Class_4.append(data[i,[1,3]])


for i in range(len(data)):
    if data[i,-1] == 0:
        Class_0.append(data[i,1:-1])
    elif data[i,-1] == 1:
        Class_1.append(data[i,1:-1])
    elif data[i,-1] == 2:
        Class_2.append(data[i,1:-1])
    elif data[i,-1] == 3:
        Class_3.append(data[i,1:-1])
    elif data[i,-1] == 4:
        Class_4.append(data[i,1:-1])

#初始数据折线图
# plt.figure()
# plt.title('Original Data')
# plt.plot(['Class0','Class1','Class2','Class3','Class4',],[len(Class_0),len(Class_1),len(Class_2),len(Class_3),len(Class_4)],color='blue',linewidth=3.0)
# plt.show()

#PCA
# plt.figure()
# plt.title('PCA')
# pca=PCA(n_components=2)
# dataset=np.concatenate([Class_0,Class_1,Class_2,Class_3,Class_4],axis=0)
# scalar=StandardScaler()
# dataset=scalar.fit_transform(dataset)
# pca1=pca.fit_transform(dataset)
# for i in range(len(Class_0)):
#     plt.scatter(pca1[i][0], pca1[i][1], alpha=0.6, color='red')
# for i in range(len(Class_0),len(Class_0)+len(Class_1),1):
#     plt.scatter(pca1[i][0], pca1[i][1], alpha=0.6, color='green')
# for i in range(len(Class_0)+len(Class_1),len(Class_0)+len(Class_1)+len(Class_2),1):
#     plt.scatter(pca1[i][0], pca1[i][1], alpha=0.6, color='blue')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)),1:
#     plt.scatter(pca1[i][0], pca1[i][1], alpha=0.6, color='orange')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)+len(Class_4),1):
#     plt.scatter(pca1[i][0], pca1[i][1], alpha=0.6, color='brown')
# plt.show()

#PCA (减少数据版）
# plt.figure()
# plt.title('PCA change')
# pca=PCA(n_components=2)
# dataset=np.concatenate([Class_0,Class_1,Class_2,Class_3,Class_4],axis=0)
# scalar=StandardScaler()
# dataset=scalar.fit_transform(dataset)
# pca1=pca.fit_transform(dataset)
# for i in range(len(Class_0)):
#     plt.scatter(pca1[i][0], pca1[i][1], alpha=0.6, color='red')
# for i in range(len(Class_0),len(Class_0)+len(Class_1),1):
#     plt.scatter(pca1[i][0], pca1[i][1], alpha=0.6, color='green')
# for i in range(len(Class_0)+len(Class_1),len(Class_0)+len(Class_1)+len(Class_2),1):
#     plt.scatter(pca1[i][0], pca1[i][1], alpha=0.6, color='blue')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)),1:
#     plt.scatter(pca1[i][0], pca1[i][1], alpha=0.6, color='orange')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)+len(Class_4),1):
#     plt.scatter(pca1[i][0], pca1[i][1], alpha=0.6, color='brown')
# plt.show()

#T-SNE
# plt.figure()
# plt.title('T-SNE')
# tsne=TSNE(n_components=2,perplexity=20)
# dataset=np.concatenate([Class_0,Class_1,Class_2,Class_3,Class_4],axis=0)
# scaler=StandardScaler()
# dataset=scaler.fit_transform(dataset)
# tsne.fit_transform(dataset)
# newdataset=tsne.embedding_
# for i in range(len(Class_0)):
#     plt.scatter(newdataset[i][0], newdataset[i][1], alpha=0.5, color='red')
# for i in range(len(Class_0),len(Class_0)+len(Class_1),1):
#     plt.scatter(newdataset[i][0], newdataset[i][1], alpha=0.5, color='green')
# for i in range(len(Class_0)+len(Class_1),len(Class_0)+len(Class_1)+len(Class_2),1):
#     plt.scatter(newdataset[i][0], newdataset[i][1], alpha=0.5, color='blue')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)),1:
#     plt.scatter(newdataset[i][0], newdataset[i][1], alpha=0.5, color='orange')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)+len(Class_4),1):
#     plt.scatter(newdataset[i][0], newdataset[i][1], alpha=0.5, color='brown')
# plt.show()

#NMF
# plt.figure()
# plt.title('NMF')
# nmf = NMF(n_components=2)         #定义维度
# dataset = np.concatenate([Class_0,Class_1,Class_2,Class_3,Class_4],axis=0)     #拼接数组
# scaler = minmax_scale(dataset)      #非负数组进行标准化
# nmf1 = nmf.fit_transform(scaler)
# for i in range(len(Class_0)):
#     plt.scatter(nmf1[i][0], nmf1[i][1], alpha=0.5, color='red')
# for i in range(len(Class_0),len(Class_0)+len(Class_1),1):
#     plt.scatter(nmf1[i][0], nmf1[i][1], alpha=0.5, color='green')
# for i in range(len(Class_0)+len(Class_1),len(Class_0)+len(Class_1)+len(Class_2),1):
#     plt.scatter(nmf1[i][0], nmf1[i][1], alpha=0.5, color='blue')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)),1:
#     plt.scatter(nmf1[i][0], nmf1[i][1], alpha=0.5, color='orange')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)+len(Class_4),1):
#     plt.scatter(nmf1[i][0], nmf1[i][1], alpha=0.5, color='brown')
# plt.show()

#LDA
# plt.figure()
# plt.title('LDA')
# lda = LDA(n_components=2)
# train_x = np.concatenate([Class_0,Class_1,Class_2,Class_3,Class_4],axis=0)
# scaler = StandardScaler()       #数据进行标准化
# train_x = scaler.fit_transform(train_x)
# Class_0_y = np.zeros(len(Class_0))        #给对应数据赋标签
# Class_1_y = np.zeros(len(Class_1))+1
# Class_2_y = np.zeros(len(Class_2))+2
# Class_3_y = np.zeros(len(Class_3))+3
# Class_4_y = np.zeros(len(Class_4))+4
# train_y = np.concatenate([Class_0_y,Class_1_y,Class_2_y,Class_3_y,Class_4_y],axis=0)
# afterlda = lda.fit_transform(train_x,train_y)
# for i in range(len(Class_0)):
#     plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, color='red')
# for i in range(len(Class_0),len(Class_0)+len(Class_1),1):
#     plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, color='green')
# for i in range(len(Class_0)+len(Class_1),len(Class_0)+len(Class_1)+len(Class_2),1):
#     plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, color='blue')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)),1:
#     plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, color='orange')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)+len(Class_4),1):
#     plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, color='brown')
# plt.show()

#标准差
# plt.figure()
# plt.title('STD')
# plt.scatter(np.array(Class_0).std(-1),np.zeros(len(Class_0)))
# plt.scatter(np.array(Class_1).std(-1),np.zeros(len(Class_1))+1)
# plt.scatter(np.array(Class_2).std(-1),np.zeros(len(Class_2))+2)
# plt.scatter(np.array(Class_3).std(-1),np.zeros(len(Class_3))+3)
# plt.scatter(np.array(Class_4).std(-1),np.zeros(len(Class_4))+4)
# plt.show()


##分类器部分

#KNN
#数据预处理
# accuracy=[]
# for x in range(1,11,1):
#     train_set, test=train_test_split(data[:,1:7],test_size=0.3,random_state=42)  #训练集占70%，测试集占30%.
#     #将随机生成的训练集和测试集中样本特征与标签分开
#     X_train=train_set[:,0:5]
#     X_test=test[:,0:5]
#     Y_train=train_set[:,5]
#     Y_test=test[:,5]
#     #将训练集和测试集中的样本特征数据标准化
#     sc=StandardScaler()
#     sc.fit(X_train)
#     X_train1=sc.transform(X_train)
#     X_test1=sc.transform(X_test)
#
#     #创建KNN分类器
#     for i in range(1,10):
#        clf = KNeighborsClassifier(n_neighbors=i,weights='distance')
#        #训练数据
#        clf.fit(X_train1,Y_train)
#        #测试数据
#        test_predict=clf.predict(X_test1)
#        accuracy.append(accuracy_score(Y_test,test_predict))
#        print('Accuracy:',accuracy[x-1])  #输出准确度
# #预测数据绘图
# plt.figure()
# plt.title('KNN Accuracy')
# plt.plot(['1','2','3','4','5','6','7','8','9','10'],[accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7],accuracy[8],accuracy[9]],color='red',linewidth=3.0)
# plt.show()

#随机森林
#数据处理（和KNN一致）
# accuracy1=[]
# for x in range(1,11,1):
#     train_set, test=train_test_split(data[:,1:7],test_size=0.3)
#     X_train=train_set[:,0:5]
#     X_test=test[:,0:5]
#     Y_train=train_set[:,5]
#     Y_test=test[:,5]
#     model = RandomForestClassifier(max_depth=x,n_estimators=1000,random_state=42)   #max_depth表示树的深度
#     model.fit(minmax_scale(X_train),Y_train)
#     score=model.score(minmax_scale(X_test),Y_test)  #得出准确度
#     accuracy1.append(score)
#     print(score)
# #随机森林结果可视化
# plt.figure()
# plt.title('Random Forest')
# plt.plot(['1','2','3','4','5','6','7','8','9','10'],[accuracy1[0],accuracy1[1],accuracy1[2],accuracy1[3],accuracy1[4],accuracy1[5],accuracy1[6],accuracy1[7],accuracy1[8],accuracy1[9]],color='blue',linewidth=3.0)
# plt.show()



