#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import sys

import numpy as np

from PyAstronomy import pyasl

from astropy.io import fits

import matplotlib.pyplot as plt

from pyPLATEFIT.platefit_init import platefit_init
from pyPLATEFIT.platefit_continfit import platefit_continfit


cspeed = 2.99792E5

settings = platefit_init()

data_dir = 'PLATEFIT_testdata/'
name = 'udf_udf10_00010.fits'  # 'udf_mosaic_01011_newmask.fits'

cat = fits.open(data_dir + name, mode='denywrite', memmap=True,
                do_not_scale_image_data=True)
flux_orig = cat[7].data[:]
err_orig  = np.sqrt(cat[8].data[:])


n_pixelmodel = cat[7].header['NAXIS1']
wl0 = cat[7].header['CRVAL1']
dw = cat[7].header['CDELT1']

settings['channel_width'] = dw

l = np.arange(n_pixelmodel) * dw + wl0

vdisp = 80.0
z = 0.27494  # 1.03622


# read information into the settings...
settings['z'] = z
settings['wavelength_range_for_fit'] = [4750./(1.0+z), 9300./(1.0+z)]
settings['vdisp'] = vdisp

# Interpolate onto uniform log-lambda scale
# The wavelength array is in air
airwl_orig = np.array(l[:])
vacwl_orig = np.array(airwl_orig[:])
vacwl_orig = pyasl.airtovac(vacwl_orig)

tmp_logwl = np.log10(vacwl_orig)

dw = np.min(np.abs(tmp_logwl - np.roll(tmp_logwl, 1)))
if (dw==0).any():
    sys.exit('ABORT: The wavelengths are identical somewhere!')


# Shift wavelengths of the spectrum from air to vacuum. From now on, unless the wavelength is AIRWL,
# the wavelengths below are in vacuum.

xnew = np.arange(np.min(tmp_logwl), np.max(tmp_logwl), dw)

use = np.where((err_orig > 0.0) & (err_orig < 1.0E5))
ynew = np.interp(10**xnew, vacwl_orig[use], flux_orig[use])
errnew = np.interp(10**xnew, vacwl_orig[use], err_orig[use])

# cut_wave = np.array(np.where((l > 4600.) & (l < 9300.))).squeeze()

vacwl = 10.0**xnew
logwl = xnew
flux = ynew
err = errnew
airwl = 10.0**xnew
airwl = pyasl.vactoair(airwl)

restwl = airwl/(1.0+settings['z'])

ok = np.array(
    np.where((err > 0.0) & (np.isfinite(err**2) == True) & (err < 1.0E10) & (np.isfinite(flux) == True)) ).squeeze()


# IMPORTANT NOTE:
#       At this point the fluxes and errors need to be de-redden and adjust for (1+z) scaling. Data is NOT corrected
#       for the forground extinction corrections. This functionality to be added later...
#


# scale by (1+z)
flux = flux * (1.0 + z)
err = err * (1.0 + z)

# ----------------------------------------------------------------------------------------------------------------------
#                                          Fit continuum using NNLS
# ----------------------------------------------------------------------------------------------------------------------
# set 'debug=True' for debug plots
best_continuum, settings = platefit_continfit(logwl[ok], restwl[ok], flux[ok], err[ok], settings, debug=True)


plt.figure()

plt.plot(restwl, flux, 'k-')  #  spectrum
plt.plot(restwl, best_continuum, 'g-')  # best fit
# plt.plot(restwl, flux - best_continuum, '-', color='y', lw=0.5)  # residuals

plt.show()
del restwl, best_continuum
