"""
Corrected Spectra
-----------------
The script examples/datasets/compute_sdss_pca.py uses an iterative PCA
technique to reconstruct masked regions of SDSS spectra.  Several of the
resulting spectra are shown below.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure is an example from astroML: see http://astroML.github.com
import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import sdss_corrected_spectra
import scipy.io as sio
import pylab as pl

#------------------------------------------------------------
# Fetch the data
data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()
spectra = sdss_corrected_spectra.reconstruct_spectra(data)
lam = sdss_corrected_spectra.compute_wavelengths(data)

Y = data['lineindex_cln']

#------------------------------------------------------------
# Plot several spectra
fig = plt.figure(figsize=(8, 8))

fig.subplots_adjust(hspace=0)

for i in range(5):
    ax = fig.add_subplot(511 + i)
    ax.plot(lam, spectra[i], '-k')

    if i < 4:
        ax.xaxis.set_major_formatter(plt.NullFormatter())
    else:
        ax.set_xlabel('wavelength $(\AA)$')

    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_ylabel('flux')

#plt.show()

pl.ion()
pl.figure()

for i_class in (2, 3, 4, 5, 6):
    i = np.where(Y == i_class)[0][0]
    l = pl.plot(lam, spectra[i] + 20 * i_class)
    c = l[0].get_color()

pl.subplots_adjust(hspace=0)
pl.xlabel('wavelength (Angstroms)')
pl.ylabel('flux + offset')
pl.title('Sample of Spectra')

#pl.show()

#------------------------------------------------------------
# Inspect the data
M = len(spectra[1])
N = len(spectra)
print "Num features: ",M
print "Num samples: ",N
#print lam

#------------------------------------------------------------
# Convert to MATLAB

mu = data['mu']

sio.savemat('spectraInputs.mat',dict(x=range(N),s=spectra,mu=mu))
sio.savemat('spectraOutputs.mat',dict(x=range(N),t=Y))
sio.savemat('spectraWavelenght.mat', dict(x=range(M),w=lam))

input("Press Enter to continue...")
