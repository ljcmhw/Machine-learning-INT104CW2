
function [bestacc,bestc,bestg]=SVMcg(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)



if nargin < 10^(-4)
    accstep = 4.5;
end
if nargin < 8
    cstep = 0.8;
    gstep = 0.8;
end
if nargin < 7
    v = 5;
end
if nargin < 5
    gmax = 8;
    gmin = -8;
end
if nargin < 3
    cmax = 8;
    cmin = -8;
end

%X=c,Y=g,cg=accuracy
[X,Y]=meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n]=size(X);
cg=zeros(m,n);
eps=10^(-4);%容忍度

%遍历不同的cg组合，找出c最小且accuracy最大的
bestc=1;
bestg=0.1;
bestacc=0;
basenum = 2;
for i=1:m
    for j=1:n
        cmd = ['-v ',num2str(v),' -c ',num2str(basenum^X(i,j)),' -g ',num2str(basenum^Y(i,j))];%-v后面一定要有一个空格！-c和-g的前后都要有空格！
        cg(i,j) = libsvmtrain(train_label, train, cmd);
        
        
%         if cg(i,j) <= 55
%             continue;
%         end
            
        if cg(i,j) >bestacc
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        
        %避免获得由于过拟合导致的过高的accuracy
         if abs(cg(i,j)-bestacc)<=eps&&bestc>basenum^X(i,j)
             bestacc = cg(i,j);
             bestc = basenum^X(i.j);
             bestg = basenum^Y(i,j);
         end
    end
end
        
%SVM超参c和g的选择情况图
figure;
meshc(X,Y,cg);
axis([cmin,cmax,gmin,gmax,30,100]);
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
zlabel('Accuracy(%)','FontSize',12);
firstline='SVC参数选择结果3D图（GridSearchMethod)';
secondline=['Best c=',num2str(bestc),' g=',num2str(bestg),' Accuracy=',num2str(bestacc),'%'];
title({firstline;secondline},'Fontsize',12);




