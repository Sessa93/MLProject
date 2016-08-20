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
[best_c, best_gamma, best_d, ~] = train_svm_poly(double(train_x),double(train_t));
baseline_poly = libsvmtrain(double(train_t),double(train_x), sprintf('-t 1 -c %f -g %f -d %f -q', best_c, best_gamma, best_d));
[predicted_label] = libsvmpredict(double(test_t), double(test_x), baseline_poly, '-q');
baseline_acc_poly = sum(predicted_label == test_t)/n_test;

%Best Degree: 4

%% Detailed Poly XVal
[best_c, best_gamma,~] = train_svm(double(train_x),double(train_t));
baseline_poly = libsvmtrain(double(train_t),double(train_x), sprintf('-t 1 -c %f -g %f -d %f -q', best_c, best_gamma, 3));
[predicted_label] = libsvmpredict(double(test_t), double(test_x), baseline_poly, '-q');
baseline_acc_poly = sum(predicted_label == test_t)/n_test;
[base_p, base_r] = calculate_metrics(predicted_label,double(test_t));




