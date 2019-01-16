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
# The example also shows how to include a library of templates
# and how to mask gas emission lines if present.
#
# MODIFICATION HISTORY:
#   V1.0.0: Written by Michele Cappellari, Leiden 11 November 2003
#   V1.1.0: Log rebin the galaxy spectrum. Show how to correct the velocity
#       for the difference in starting wavelength of galaxy and templates.
#       MC, Vicenza, 28 December 2004
#   V1.1.1: Included explanation of correction for instrumental resolution.
#       After feedback from David Valls-Gabaud. MC, Venezia, 27 June 2005
#   V2.0.0: Included example routine to determine the goodPixels vector
#       by masking known gas emission lines. MC, Oxford, 30 October 2008
#   V2.0.1: Included instructions for high-redshift usage. Thanks to Paul Westoby
#       for useful feedback on this issue. MC, Oxford, 27 November 2008
#   V2.0.2: Included example for obtaining the best-fitting redshift.
#       MC, Oxford, 14 April 2009
#   V2.1.0: Bug fix: Force PSF_GAUSSIAN to produce a Gaussian with an odd
#       number of elements centered on the middle one. Many thanks to
#       Harald Kuntschner, Eric Emsellem, Anne-Marie Weijmans and
#       Richard McDermid for reporting problems with small offsets
#       in systemic velocity. MC, Oxford, 15 February 2010
#   V2.1.1: Added normalization of galaxy spectrum to avoid numerical
#       instabilities. After feedback from Andrea Cardullo.
#       MC, Oxford, 17 March 2010
#   V2.2.0: Perform templates convolution in linear wavelength.
#       This is useful for spectra with large wavelength range.
#       MC, Oxford, 25 March 2010
#   V2.2.1: Updated for Coyote Graphics. MC, Oxford, 11 October 2011
#   V2.2.2: Renamed PPXF_KINEMATICS_EXAMPLE_SAURON to avoid conflict with the
#       new PPXF_KINEMATICS_EXAMPLE_SDSS. Removed DETERMINE_GOOPIXELS which was
#       made a separate routine. MC, Oxford, 12 January 2012
#   V3.0.0: Translated from IDL into Python. MC, Oxford, 6 December 2013
#   V3.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
#   V3.0.2: Explicitly sort template files as glob() output may not be sorted.
#       Thanks to Marina Trevisan for reporting problems under Linux.
#       MC, Sydney, 4 February 2015
#   V3.0.3: Use redshift in determine_goodpixels. MC, Oxford, 5 May 2015
#   V3.0.4: Support both Pyfits and Astropy to read FITS files.
#       MC, Oxford, 22 October 2015
#   V3.1.0: Replaced the Vazdekis-99 SSP models with the Vazdekis+10 ones.
#       MC, Oxford, 3 May 2016
#   V3.1.1: Illustrate how to use a higher sampling for the template, via the
#       `velscale_ratio` keyword. MC, Oxford, 22 June 2016
#   V3.1.2: Use proper input `vsyst` when `velscale_ratio` > 1.
#       Thanks to Kyle Westfall for finding this issue in the MaNGA DAP.
#       MC, Oxford, 19 October 2016
#   V3.1.3: Make files paths relative to this file, to run the example from
#       any directory. MC, Oxford, 18 January 2017
#
##############################################################################

from __future__ import print_function

import glob
from os import path
from time import clock

from astropy.io import fits
from scipy import ndimage
import numpy as np

from ppxf import ppxf
import ppxf_util as util

def ppxf_example_kinematics_sauron():

    file_dir = path.dirname(path.realpath(__file__))  # path of this procedure

    # Read a galaxy spectrum and define the wavelength range
    #
    file = file_dir + '/spectra/NGC4550_SAURON.fits'
    hdu = fits.open(file)
    gal_lin = hdu[0].data
    h1 = hdu[0].header

    lamRange1 = h1['CRVAL1'] + np.array([0., h1['CDELT1']*(h1['NAXIS1'] - 1)])
    FWHM_gal = 4.2  # SAURON has an instrumental resolution FWHM of 4.2A.

    # If the galaxy is at a significant redshift (z > 0.03), one would need to apply
    # a large velocity shift in PPXF to match the template to the galaxy spectrum.
    # This would require a large initial value for the velocity (V > 1e4 km/s)
    # in the input parameter START = [V,sig]. This can cause PPXF to stop!
    # The solution consists of bringing the galaxy spectrum roughly to the
    # rest-frame wavelength, before calling PPXF. In practice there is no
    # need to modify the spectrum before the usual LOG_REBIN, given that a
    # red shift corresponds to a linear shift of the log-rebinned spectrum.
    # One just needs to compute the wavelength range in the rest-frame
    # and adjust the instrumental resolution of the galaxy observations.
    # This is done with the following three commented lines:
    #
    # z = 1.23 # Initial estimate of the galaxy redshift
    # lamRange1 = lamRange1/(1+z) # Compute approximate restframe wavelength range
    # FWHM_gal = FWHM_gal/(1+z)   # Adjust resolution in Angstrom

    galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_lin)
    galaxy = galaxy/np.median(galaxy)  # Normalize spectrum to avoid numerical issues
    noise = np.full_like(galaxy, 0.0047)           # Assume constant noise per pixel here

    # Read the list of filenames from the Single Stellar Population library
    # by Vazdekis (2010, MNRAS, 404, 1639) http://miles.iac.es/. A subset
    # of the library is included for this example with permission
    vazdekis = glob.glob(file_dir + '/miles_models/Mun1.30Z*.fits')
    FWHM_tem = 2.51  # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
    velscale_ratio = 2  # adopts 2x higher spectral sampling for templates than for galaxy

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to a velocity scale 2x smaller than the SAURON galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = fits.open(vazdekis[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)])
    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale/velscale_ratio)
    templates = np.empty((sspNew.size, len(vazdekis)))

    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SAURON and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels Vazdekis --> SAURON
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in pixels

    for j, file in enumerate(vazdekis):
        hdu = fits.open(file)
        ssp = hdu[0].data
        ssp = ndimage.gaussian_filter1d(ssp, sigma)
        sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale/velscale_ratio)
        templates[:, j] = sspNew/np.median(sspNew)  # Normalizes templates

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below (see above).
    #
    c = 299792.458
    if velscale_ratio > 1:
        dv = (np.mean(logLam2[:velscale_ratio]) - logLam1[0])*c  # km/s
    else:
        dv = (logLam2[0] - logLam1[0])*c  # km/s

    vel = 450.  # Initial estimate of the galaxy velocity in km/s
    z = np.exp(vel/c) - 1   # Relation between velocity and redshift in pPXF
    goodPixels = util.determine_goodpixels(logLam1, lamRange2, z)

    # Here the actual fit starts. The best fit is plotted on the screen.
    # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
    #
    start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
    t = clock()

    pp = ppxf(templates, galaxy, noise, velscale, start,
              goodpixels=goodPixels, plot=True, moments=4,
              degree=4, vsyst=dv, velscale_ratio=velscale_ratio)

    print("Formal errors:")
    print("     dV    dsigma   dh3      dh4")
    print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

    print('Elapsed time in pPXF: %.2f s' % (clock() - t))

    # If the galaxy is at significant redshift z and the wavelength has been
    # de-redshifted with the three lines "z = 1.23..." near the beginning of
    # this procedure, the best-fitting redshift is now given by the following
    # commented line (equation 2 of Cappellari et al. 2009, ApJ, 704, L34;
    # http://adsabs.harvard.edu/abs/2009ApJ...704L..34C)
    #
    #print, 'Best-fitting redshift z:', (z + 1)*(1 + sol[0]/c) - 1

#------------------------------------------------------------------------------

if __name__ == '__main__':
    ppxf_example_kinematics_sauron()
    import matplotlib.pyplot as plt
    plt.pause(1)
