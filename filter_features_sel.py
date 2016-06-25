#Author Andrea Sessa
#Perform a filter feature selection

import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import sdss_corrected_spectra
import scipy.io as sio
import pylab as pl
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

def selectF(X,Y):
    #Load the data
    #data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()
    #X = sdss_corrected_spectra.reconstruct_spectra(data)
    #Y = data['lineindex_cln']

    X_indices = np.arange(X.shape[-1])

    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X, Y)

    #print(selector.pvalues_)

    #scores = -np.log10(selector.pvalues_)
    #scores /= scores.max()

    #plt.bar(X_indices - .45, scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)', color='g')
    #plt.show()

    return selector.transform(X).shape
