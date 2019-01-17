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
# This procedure illustrates (and tests) both
#   1. The simultaneous fit of two reflection-symmetric LOSVDs;
#   2. The fit of the sky spectrum together with the templates.
#
# An application of this approach is presented in Section 3 of
#   Weijmans, A-M et al., 2009, MNRAS, 398, 561
#   http://adsabs.harvard.edu/abs/2009MNRAS.398..561W
#
# MODIFICATION HISTORY:
#   V1.0.0: Written by Michele Cappellari, based on a previous IDL procedure.
#       Oxford, 20 April 2017
#
################################################################################

from __future__ import print_function

import glob
from os import path
from time import clock

from astropy.io import fits
import numpy as np
from scipy import signal
from numpy.polynomial import legendre
import matplotlib.pyplot as plt

from ppxf import ppxf
import ppxf_util as util

################################################################################

def ppxf_example_sky_and_symmetric_losvd():

    np.random.seed(124)  # for reproducible results

    file_dir = path.dirname(path.realpath(__file__))  # path of this procedure

    # Solar metallicity, Age=12.59 Gyr
    hdu = fits.open(file_dir + '/miles_models/Mun1.30Zp0.00T12.5893.fits')
    ssp = hdu[0].data
    h = hdu[0].header

    lamRange = h['CRVAL1'] + np.array([0.,h['CDELT1']*(h['NAXIS1']-1)])
    velscale = 70.  # km/s
    star, logLam, velscale = util.log_rebin(lamRange, ssp, velscale=velscale)
    star /= np.mean(star)

    # Adopted input parameters =================================================

    vel = 200./velscale    # Velocity of 1st spectrum in pixels (2nd has -vel)
    sigma = 300./velscale   # Dispersion of both spectra in pixels
    h3 = 0.1                # h3 of 1st spectrum (2nd has -h3)
    h4 = 0.1
    sn = 40.
    moments = 4
    deg = 4
    vshift = 10                 # Adopted systemic velocity in pixels
    vsyst = vshift*velscale     # Adopted systemic velocity in km/s

    # Generate input Sky =======================================================
    # For illustration, the sky is modelled as two Gaussian emission lines

    n = star.size
    x = np.arange(n)
    sky1 = np.exp(-0.5*(x - 1000)**2/100)
    sky2 = np.exp(-0.5*(x - 2000)**2/100)

    # Generate input LOSVD =====================================================

    dx = int(abs(vel) + 5*sigma)
    v = np.linspace(-dx, dx, 2*dx + 1)
    w = (v - vel)/sigma
    w2 = w**2
    gauss = np.exp(-0.5*w2)
    gauss /= np.sum(gauss)
    h3poly = w*(2*w2 - 3)/np.sqrt(3)
    h4poly = (w2*(4*w2 - 12) + 3)/np.sqrt(24)
    losvd = gauss*(1 + h3*h3poly + h4*h4poly)

    # Generate first synthetic spectrum ========================================
    # The template is convolved with the LOSVD

    x = np.linspace(-1, 1, n)
    galaxy1 = signal.fftconvolve(star, losvd, mode="same")
    galaxy1 = np.roll(galaxy1, vshift)   # Mimic nonzero systemic velocity
    galaxy1 *= legendre.legval(x, np.append(1, np.random.uniform(-0.1, 0.1, deg-1)))  # Multiplicative polynomials
    galaxy1 += legendre.legval(x, np.random.uniform(-0.1, 0.1, deg))  # Additive polynomials
    galaxy1 += sky1 + 2*sky2    # Add two sky lines
    galaxy1 = np.random.normal(galaxy1, 1/sn)   # Add noise

    # Generate symmetric synthetic spectrum ====================================
    # The same template is convolved with a reversed LOSVD
    # and different polynomials and sky lines are included

    galaxy2 = signal.fftconvolve(star, np.flip(losvd, 0), mode="same")
    galaxy2 = np.roll(galaxy2, vshift)   # Mimic nonzero systemic velocity
    galaxy2 *= legendre.legval(x, np.append(1, np.random.uniform(-0.1, 0.1, deg-1)))  # Multiplicative polynomials
    galaxy2 += legendre.legval(x, np.random.uniform(-0.1, 0.1, deg))  # Additive polynomials
    galaxy2 += 2*sky1 + sky2    # Add two sky lines
    galaxy2 = np.random.normal(galaxy2, 1/sn)   # Add noise

    # Load spectral templates ==================================================

    vazdekis = glob.glob(file_dir + '/miles_models/Mun1.30Z*.fits')
    templates = np.empty((n, len(vazdekis)))
    for j, file in enumerate(vazdekis):
        hdu = fits.open(file)
        ssp = hdu[0].data
        sspNew, logLam2, velscale = util.log_rebin(lamRange, ssp, velscale=velscale)
        templates[:, j] = sspNew/np.median(sspNew)  # Normalize templates

    # Do the fit ===============================================================

    # Input both galaxy spectra simultaneously to pPXF
    galaxy = np.column_stack([galaxy1, galaxy2])

    # Use two sky templates for each galaxy spectrum
    sky = np.column_stack([sky1, sky2])

    # Randomized starting guess
    vel0 = vel + np.random.uniform(-1, 1)
    sigma0 = sigma*np.random.uniform(0.8, 1.2)
    start = np.array([vel0, sigma0])*velscale  # Convert to km/s
    goodpixels = np.arange(50, n - 50)

    print("\nThe input values are: Vel=%0.0f, sigma=%0.0f, h3=%0.1f, h4=%0.1f\n" %
          (vel*velscale, sigma*velscale, h3, h4))

    t = clock()

    pp = ppxf(templates, galaxy, np.full_like(galaxy, 1/sn), velscale, start,
              goodpixels=goodpixels, plot=1, moments=moments,
              vsyst=vsyst, mdegree=deg, degree=deg, sky=sky)

    print('Elapsed time in pPXF: %.2f s' % (clock() - t))
    plt.pause(1)

################################################################################

if __name__ == '__main__':
    ppxf_example_sky_and_symmetric_losvd()
