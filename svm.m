%支持向量机


addpath ('/Users/ljcmhw/Desktop/INT104/CW2/libsvm-3.25/matlab');
filename='/Users/ljcmhw/Desktop/INT104/cw2/CW_Data.csv';
data=readmatrix(filename);
features=data(:,2:6);
class=data(:,7);
class1=class(1:360,:);
class2=class(361:end,:);
% temp=mapminmax(features',0,1);%归一化到0和1之间
% featuresn=temp';%转置为列向量
featuresn1=features(1:360,:);%取70%的特征作为训练集
featuresn2=features(361:end,:);%剩下的作为测试集
%构建svm
[bestacc,bestc,bestg] = SVMcg(class1,featuresn1,-8,8,-8,8,3,1,1,4.5);%进行交叉验证
cmd=['-c ',num2str(bestc), ' -g ',num2str(bestg),'-s 0 -t 2'];%-c是代价函数的参数;-g对应高斯核函数的sigma;-s对应于预测方式
% cmd=['-c 1 -g 100 -s 0 -t 2'];%人为设置固定参数
model=libsvmtrain(class1,featuresn1,cmd);%对数据进行训练
[predict_class,accuracy,prob_estimates]=libsvmpredict(class2,featuresn2,model,'-b probability_estimates');
%预测结果图
figure
plot(class,'bo');
hold on;
plot(predict_class,'r*');
grid on;
xlabel('Student number');
ylabel('Programme');
legend('Actual','Predict');
set(gca,'fontsize',12);

