% Prepare the workspace
clear all;
close all;
clc;

%% Load the SDSS Spectra dataset
load spectraInputs.mat
load spectraOutputs.mat
load spectraWavelenght.mat
t = t';
norm = zscore(s);

labels = ['unknown' 'star' 'absorption galaxy' 'galaxy' 'emission galaxy' 'narrow-line QSO' 'broad-line QSO' 'sky' 'Hi-z QSO' 'Late-type star'];

%% Plot sample spectrum


figure()

e = find(t == 4,1);
plot(w',s(e,:),'r')
xlabel('Wavelength(angstrom)')
ylabel('Flux')

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




