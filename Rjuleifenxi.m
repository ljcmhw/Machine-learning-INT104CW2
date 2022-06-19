\\聚类分析


filename='/Users/ljcmhw/Desktop/INT104/cw2/excel数据筛选/cw_file4.csv';
a=readmatrix(filename);
b=zscore(a(:,2:6));
r=corrcoef(a(:,2:6));
z=linkage(b','average','correlation');
h=dendrogram(z);
set(h,'Color','k','LineWidth',1.3);
T=cluster(z,'maxclust',5);
for i=1:5
  tm=find(T==i);
  fprintf('第%d类的有%s\n',i,int2str(tm'));
end
