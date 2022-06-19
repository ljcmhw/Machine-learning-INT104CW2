import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MinMaxScaler

#将原来的csv格式数据集变成神经网络使用的数据集
class CW2_dataset(Dataset):
    def __init__(self,train=True,fold=None):
        super(CW2_dataset, self).__init__()
        self.x, self.y = get_data(train,fold)   #fold表示把第fold组数据作为测试数据，其它数据放入train中

    def __getitem__(self, item):
        return self.x[item],self.y[item]

    def __len__(self):
        return len(self.x)

#用来打乱数据的函数
def shuffle_data(x,y):
    index = torch.randperm(x.size(0))
    x = x[index]
    y = y[index]
    return x,y

#getdata读取数据
def get_data(train,fold):
    df = pd.read_csv('/Users/ljcmhw/Desktop/INT104/cw2/CW_Data.csv',sep=',',header=0)
    data = df.values
    Class_0 = []
    Class_1 = []
    Class_2 = []
    Class_3 = []
    Class_4 = []
    for i in range(len(data)):
        if data[i,-1]==0:
            Class_0.append(data[i,1:-1])
        elif data[i,-1] == 1:
            Class_1.append(data[i,1:-1])
        elif data[i,-1] == 2:
            Class_2.append(data[i,1:-1])
        elif data[i,-1] == 3:
            Class_3.append(data[i,1:-1])
        elif data[i,-1] == 4:
            Class_4.append(data[i,1:-1])
    train_x = np.concatenate([Class_0,Class_1,Class_2,Class_3,Class_4],axis=0)
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    train_x = train_x.astype(np.float32)
    Class_0_y = np.zeros(len(Class_0),dtype=np.compat.long)
    Class_1_y = np.zeros(len(Class_1),dtype=np.compat.long)+1
    Class_2_y = np.zeros(len(Class_2),dtype=np.compat.long)+2
    Class_3_y = np.zeros(len(Class_3),dtype=np.compat.long)+3
    Class_4_y = np.zeros(len(Class_4),dtype=np.compat.long)+4
    train_y = np.concatenate([Class_0_y,Class_1_y,Class_2_y,Class_3_y,Class_4_y],axis=0)
    train_x,train_y = shuffle_data(torch.tensor(train_x),torch.tensor(train_y))      #对数据进行打乱方便交叉验证的合理性

    #将数据集分成20%的测试集和80%的训练集
    if fold==4:
        test_x = train_x[-int(len(train_x)/5):,:]
        test_y = train_y[-int(len(train_x)/5):]
        train_x = train_x[:-int(len(train_x)/5),:]
        train_y = train_y[:-int(len(train_x)/5)]
    elif fold==0:
        test_x = train_x[:int(len(train_x)/5),:]
        test_y = train_y[:int(len(train_x)/5)]
        train_x = train_x[int(len(train_x)/5):,:]
        train_y = train_y[int(len(train_x)/5):]
    else:
        test_x = train_x[fold*int(len(train_x)/5):(fold+1)*int(len(train_x)/5),:]
        test_y = train_y[fold*int(len(train_x)/5):(fold+1)*int(len(train_x)/5)]
        train_x1 = train_x[:fold*int(len(train_x)/5),:]
        train_y1 = train_y[:fold*int(len(train_x)/5)]
        train_x2 = train_x[(fold+1)*int(len(train_x)/5):,:]
        train_y2 = train_y[(fold+1)*int(len(train_x)/5):]
        train_x = torch.cat([train_x1,train_x2],dim=0)
        train_y = torch.cat([train_y1,train_y2],dim=0)

    if train:
        return train_x,train_y
    else:
        return test_x,test_y

#神经网络模型函数
class Net(nn.Module):
    #定义每一层的内容
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(5,16)   #第一层为输入的五维数据
        self.fc2 = nn.Linear(16,128)  #第二层
        self.fc3 = nn.Linear(128,5)   #第三层降维成一组0-1队列（有五个元素对应五种programme）
    #定义前向传递，每次线性变换中都增加一个非线性变换，其代价函数参数为sigmoid
    def forword(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)   #非线性变换
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        return x


#用神经网络进行训练
def train(model,train_loader,optimizer,epoch):   #epoch用于统计当前学习到第几个
    model.train() #将模型转换到训练模式
    for batch_idx,(data,target) in enumerate(train_loader):   #enumerate表示迭代
        optimizer.zero_grad()
        output = model.forword(data)
        loss = F.cross_entropy(output,target)   #将训练得到的结果与目标结果比较
        loss.backward()    #反向传递计算梯度
        optimizer.step()   #优化器对参数调整优化
        if batch_idx % 100 == 0:     #每100次训练输出当前的准确度与误差
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,batch_idx*len(data),
                len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))

#测试训练成果
def runtest(model,test_loader,use_trainset=True):
    model.train()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model.forword(data)
            test_loss += F.cross_entropy(output,target,reduction='sum').item()
            #通过判断输出队列中最大值所在位置和标签是否一致判断训练模型准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if use_trainset:
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))
    else:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))



def main():
    #创建dataset
    fold = 1
    trainset = CW2_dataset(train=True,fold=fold)
    testset = CW2_dataset(train=False,fold=fold)
    #将数据集分割成多个小集合进行训练
    trainset_loader = DataLoader(trainset,batch_size=22,shuffle=True)
    testset_loader = DataLoader(testset,batch_size=22,shuffle=True)
    #调用神经网络模型函数Net
    model = Net()  #得到模型
    #参数优化
    optimizer = optim.Adam(model.parameters(),lr=0.9)   #lr是学习率,Adam是一种逐参数适应学习率方法
    # optimizer = optim.SGD(model.parameters(), lr=0.9,momentum=0.1,dampening=0,weight_decay=1e-4,nesterov=True)   #采用SGD+nesterov动量的方法进行优化
    scheduler = StepLR(optimizer,step_size=250,gamma=0.9)    #让学习率逐渐降低，避免无法逼近局部最优解
    for epoch in range(1,1000,1):       #1000次迭代训练
        train(model,trainset_loader,optimizer,epoch)
        #分别看在训练集和测试集上的训练成果
        runtest(model,trainset_loader,use_trainset=True)
        runtest(model,testset_loader,use_trainset=False)
        scheduler.step() #计数
        

if __name__ =='__main__':
    main()
