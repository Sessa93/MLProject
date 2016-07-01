%Author: Andrea Sessa
%Email: andrea.sessa@mail.polimi.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This script extract and visualize the important characteristics
%of the SDSS spectra dataset

%% Workspace preparation
clear all;
close all;
clc;

%% Dataset loading
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

%% Plot sample spectrum
figure()

for ii=[2,3,4,6]
    e = find(t == ii,1);

    plot(w',s(e,:)+20*ii)
    hold on
    %text(6600,20*ii+30,labels(:,ii+1))
    hold on
end
xlabel('Wavelength(angstrom)')
ylabel('Flux')

%% Plot classes distribution

figure()
histogram(t)
ax = gca;
lab = {'star', 'abs galaxy', 'galaxy', 'em. galaxy', 'narrow QSO', 'broad QSO', 'Late star'};
set(gca,'XLim',[0 8],'XTick',1:7,'XTickLabel',lab)
