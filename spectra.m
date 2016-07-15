%Author: Andrea Sessa
%Email: andrea.sessa@mail.polimi.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This script performs all the experiment described in report.pdf

%% Prepare the workspace
clear all;
close all;
clc;

%% Load the SDSS Spectra dataset
load spectraInputs.mat
load spectraOutputs.mat
load spectraWavelenght.mat
t = t';

[N,M] = size(s);

%Normalization
norm = zscore(s);

%Shuffling
indx = randperm(N);
t = t(indx);
s = s(indx,:);

%To remove
t(t == 9) = 7;
t(t == 0) = 3;

labels = { 'unknown','star', 'absorption galaxy', 'galaxy', 'emission galaxy', 'narrow-line QSO', 'broad-line QSO', 'sky', 'Hi-z QSO', 'Late-type star'};

%% Prepare the dataset
perc_train = 0.8;
perc_test = 1 - perc_train;

n_train = N*perc_train;
n_test = N*perc_test;
norm = s;

%Divide in train/test
train_x = norm(1:n_train,:);
test_x = norm(n_train+1:n_test+n_train,:);
train_t = t(1:n_train);
test_t = t(n_train+1:n_test+n_train);

%% Baseline - No FS
[best_c, best_gamma cv_acc] = train_svm(double(train_x),double(train_t));
baseline = libsvmtrain(double(train_t),double(train_x), sprintf('-t 2 -c %f -g %f -q', best_c, best_gamma));
[predicted_label] = libsvmpredict(double(test_t), double(test_x), baseline, '-q');
baseline_acc = sum(predicted_label == test_t)/n_test;

%% Determine the correct polynomial degree
[best_c, best_gamma, best_d, cv_acc] = train_svm_poly(double(train_x),double(train_t));
baseline_poly = libsvmtrain(double(train_t),double(train_x), sprintf('-t 2 -c %f -g %f -d %f -q', best_c, best_gamma, best_d));
[predicted_label] = libsvmpredict(double(test_t), double(test_x), baseline_poly, '-q');
baseline_acc_poly = sum(predicted_label == test_t)/n_test;

%% PCA Analysis

[loads,scores,var] = pca(train_x);

figure()

pc1 = scores(:,1);
pc2 = scores(:,2);
data = double([pc1 pc2]);

[best_c, best_gamma, cv_acc] = train_svm(data,double(train_t));
pca_model = libsvmtrain(double([pc1 pc2]),double(train_t), sprintf('-t 2 -c %f -g %f -q', best_c, best_gamma));
[predicted_label] = libsvmpredict(double(test_t), double(test_x), pca_model, '-q');
pca_acc = sum(predicted_label == test_t)/n_test;

%% Fw features selections

%Reduce the dataset
xt = train_x(1:2000,:);
tt = train_t(1:2000);

%5-Fold cross validation
c = cvpartition(double(tt),'k',5);
opts = statset('display','iter');

[fs,history] = sequentialfs(@svmwrapper,double(xt),double(tt),'cv',c,'options',opts,'nfeatures',20)







