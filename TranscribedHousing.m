% Note: this script requires the Statistics and Machine Learning Toolbox to
% function properly

% Download housing prices
fn='housing.txt';
hold on
% UCI download
urlwrite('http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',fn);
inputNames={'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'};
outputName={'MEDV'};
housingAttributes=[inputNames,outputName];
% Input data
formatSpec='%8f%7f%8f%3f%8f%8f%7f%8f%4f%7f%7f%7f%7f%f%[^\n\r]';
fileID=fopen(fn,'r');
dataArray=textscan(fileID,formatSpec,'Delimiter','','WhiteSpace','','ReturnOnError',false);
fclose(fileID);
% Write in data
housing=table(dataArray{1:end-1},'VariableNames',{'VarName1','VarName2','VarName3','VarName4','VarName5','VarName6','VarName7','VarName8','VarName9','VarName10','VarName11','VarName12','VarName13','VarName14'});
% Delete file, clear temp vars
clearvars fn formatSpec fileID dataArray ans;
% Delete housing.txt, read into a table
housing.Properties.VariableNames=housingAttributes;
X=housing{:,inputNames};
y=housing{:,outputName};
% Write the heatmap
hold off
heatmap(corrcoef([X y]))
X=zscore(X);
% Covariance matrix
[n,p]=size(X);
covm=cov(X);
% Compute eigenvalues and eigenvectors of covariance matrix
[eigvec,eigval]=eig(covm);
eigval=diag(eigval);
% Reformat eigenvalues and eigenvectors
eigval=flipud(eigval);
eigvec=eigvec(:,p:-1:1);
% Scree plot
f=figure;
f,plot(1:length(eigval),eigval,'ko- ')
title('Scree Plot')
xlabel('Eigvenvalue Index - k')
ylabel('Eigvenvalue')
% Explained variance percentage
hold on
pervar=100*cumsum(eigval/sum(eigval));
g=figure;
g,plot(1:13,pervar,'-bo')
% This is set from above results
k=5;
T=eigvec(:,1:k);
newX=[ones(size(X*T,1),1) X*T];
% Apply OLS on newX and y
[Q,R]=qr(newX);
% Compute the LS coefficients
beta=R\(Q'*y);
yfit=newX*beta;
residuals=y-yfit
% Stem plot
xlabel('Observations');
ylabel('Residuals');
h=figure;
h,stem(residuals)
% True and predicted values
h1=scatter((1:506),yfit)
hold on
h2=scatter((1:506),y)
xlabel('Observations');
% MSE; RMSE; mean(y)
MSE=mean(residuals.^2)
RMSE=sqrt(MSE)
mean(y)