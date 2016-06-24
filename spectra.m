% Prepare the workspace
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

labels = {'unknown', 'star', 'absorption galaxy', 'galaxy', 'emission galaxy', 'narrow-line QSO', 'broad-line QSO', 'sky', 'Hi-z QSO', 'Late-type star'};

%% Plot sample spectrum
figure()

for ii=[0,2,4,6]
    e = find(t == ii,1);

    plot(w',s(e,:)+20*ii)
    hold on
    text(6600,20*ii+30,labels(:,ii+1))
    hold on
end
xlabel('Wavelength(angstrom)')
ylabel('Flux')

%% Prepare the dataset
perc_train = 0.8;
perc_test = 1 - perc_train;

n_train = N*perc_train;
n_test = N*perc_test;

%Divide in train/test
train_x = s(1:n_train,:);
test_x = s(n_train+1:n_test+n_train,:);
train_t = t(1:n_train);
test_t = t(n_train+1:n_test+n_train);

%% Baseline - No FS
train_svm(train_x,train_t)

%% PCA Analysis

[loads,scores,var] = pca(norm);

figure()

pc1 = scores(:,1);
pc2 = scores(:,2);
pc3 = scores(:,3);

gscatter(pc1,pc3,t);

figure()
gscatter(pc2,pc3,t);

figure()
gscatter(pc1,pc2,t);

%% Fw features selections




