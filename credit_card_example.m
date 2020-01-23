clear all
data = load('germancredit_data.txt');
y = data(:,end);
y(y==2)=0;
X = data(:,1:end-1);
X = [zscore(X(:,1:15)),X(:,16:end)]; % standard continuous predictors 
%================= end data processing =============================%
rng(12345)
[mu_MGVB,Sig,LB_MGVB] = manifold_GVB_logistic(X,y);
figure
plot(LB_MGVB,'-')
title('lower bound Manifold GVB')
xlabel('Iterations')
ylabel('Lower bound')


 