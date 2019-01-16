#!/usr/bin/env python
##############################################################################
#
# Usage example for the procedure PPXF, which implements the
# Penalized Pixel-Fitting (pPXF) method originally described in
# Cappellari M., & Emsellem E., 2004, PASP, 116, 138
#     http://adsabs.harvard.edu/abs/2004PASP..116..138C
# and upgraded in Cappellari M., 2017, MNRAS, 466, 798
#     http://adsabs.harvard.edu/abs/2017MNRAS.466..798C
#
# This example shows how to fit multiple stellar components with different
# stellar population and kinematics.
#
# MODIFICATION HISTORY:
#   V1.0.0: Early test version. Michele Cappellari, Oxford, 20 July 2009
#   V1.1.0: Cleaned up for the paper by Johnston et al. (MNRAS, 2013).
#       MC, Oxford, 26 January 2012
#   V2.0.0: Converted to Python and adapted to the changes in the new public
#       PPXF version, Oxford 8 January 2014
#   V2.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
#   V2.0.2: Support both Pyfits and Astropy to read FITS files.
#       MC, Oxford, 22 October 2015
#   V2.0.3: Use proper noise in input. MC, Oxford, 8 March 2016
#   V2.1.0: Replaced the Vazdekis-99 SSP models with the Vazdekis+10 ones.
#       MC, Oxford, 3 May 2016
#   V2.1.1: Make files paths relative to this file, to run the example from
#       any directory. MC, Oxford, 18 January 2017
#
##############################################################################

from __future__ import print_function

from os import path
from time import clock

from astropy.io import fits
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from ppxf import ppxf
import ppxf_util as util

def ppxf_example_two_components():

    file_dir = path.dirname(path.realpath(__file__))  # path of this procedure

    hdu = fits.open(file_dir + '/miles_models/Mun1.30Zp0.00T12.5893.fits')  # Solar metallicitly, Age=12.59 Gyr
    gal_lin = hdu[0].data
    h1 = hdu[0].header
    lamRange1 = h1['CRVAL1'] + np.array([0., h1['CDELT1']*(h1['NAXIS1']-1)])
    c = 299792.458      # speed of light in km/s
    velscale = c*h1['CDELT1']/max(lamRange1)   # Do not degrade original velocity sampling
    model1, logLam1, velscale = util.log_rebin(lamRange1, gal_lin, velscale=velscale)
    model1 /= np.median(model1)

    hdu = fits.open(file_dir + '/miles_models/Mun1.30Zp0.00T01.0000.fits')  # Solar metallicitly, Age=1.00 Gyr
    gal_lin = hdu[0].data
    model2, logLam1, velscale = util.log_rebin(lamRange1, gal_lin, velscale=velscale)
    model2 /= np.median(model2)

    model = np.column_stack([model1, model2])
    galaxy = np.empty_like(model)

    # These are the input values in spectral pixels
    # for the (V,sigma) of the two kinematic components
    #
    vel = np.array([0., 250.])/velscale
    sigma = np.array([200., 100.])/velscale

    # The synthetic galaxy model consists of the sum of two
    # SSP spectra with age of 1Gyr and 13Gyr respectively
    # with different velocity and dispersion
    #
    for j in range(len(vel)):
        dx = int(abs(vel[j]) + 4.*sigma[j])   # Sample the Gaussian at least to vel+4*sigma
        v = np.linspace(-dx, dx, 2*dx + 1)
        losvd = np.exp(-0.5*((v - vel[j])/sigma[j])**2) # Gaussian LOSVD
        losvd /= np.sum(losvd)      # normaize LOSVD
        galaxy[:, j] = signal.fftconvolve(model[:, j], losvd, mode="same")
        galaxy[:, j] /= np.median(model[:, j])
    galaxy = np.sum(galaxy, axis=1)
    sn = 100.
    np.random.seed(2) # Ensure reproducible results
    noise = galaxy/sn
    galaxy = np.random.normal(galaxy, noise) # add noise to galaxy

    # Adopts two templates per kinematic component
    #
    templates = np.column_stack([model1, model2, model1, model2])

    # Start both kinematic components from the same guess.
    # With multiple stellar kinematic components
    # a good starting guess is essential
    #
    start = [np.mean(vel)*velscale, np.mean(sigma)*velscale]
    start = [start, start]
    goodPixels = np.arange(20, 6000)

    t = clock()

    plt.clf()
    plt.subplot(211)
    plt.title("Two components pPXF fit")
    print("+++++++++++++++++++++++++++++++++++++++++++++")

    pp = ppxf(templates, galaxy, noise, velscale, start,
              goodpixels=goodPixels, plot=True, degree=4,
              moments=[2, 2], component=[0, 0, 1, 1])

    plt.subplot(212)
    plt.title("Single component pPXF fit")
    print("---------------------------------------------")

    start = start[0]
    pp = ppxf(templates, galaxy, noise, velscale, start,
              goodpixels=goodPixels, plot=True, degree=4, moments=2)

    print("=============================================")
    print("Total elapsed time %.2f s" % (clock() - t))

    plt.tight_layout()
    plt.pause(1)


#------------------------------------------------------------------------------

if __name__ == '__main__':
    ppxf_example_two_components()
