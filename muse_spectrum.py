#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import sys

import numpy as np
from mpdaf.sdetect import Source

from PyAstronomy import pyasl

from astropy.io import fits

import matplotlib.pyplot as plt

from pyplatefit.platefit_init import platefit_init
from pyplatefit.platefit_continfit import platefit_continfit, plot_bestmodel

debug = True

cspeed = 2.99792E5

settings = platefit_init()

data_dir = 'PLATEFIT_testdata/'
name = 'udf_udf10_00010.fits'
# name = 'udf_mosaic_01011_newmask.fits.gz'
src = Source.from_file(data_dir+name)

sp = src.spectra[src.REFSPEC]
flux_orig = sp.data
err_orig  = np.sqrt(sp.var)


#n_pixelmodel = cat[7].header['NAXIS1']
#wl0 = cat[7].header['CRVAL1']
#dw = cat[7].header['CDELT1']

settings['channel_width'] = sp.wave.get_step()

l = sp.wave.coord()

vdisp = 80.0
z = src.z[src.z['Z_DESC']=='MUSE']['Z'][0]


# read information into the settings...
settings['z'] = z
settings['wavelength_range_for_fit'] = [l[0]/(1.0+z), l[-1]/(1.0+z)]
settings['vdisp'] = vdisp

# Interpolate onto uniform log-lambda scale
# The wavelength array is in air
airwl_orig = np.array(l[:])
vacwl_orig = np.array(airwl_orig[:])
vacwl_orig = pyasl.airtovac2(vacwl_orig)

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
airwl = pyasl.vactoair2(airwl)

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
best_continuum, settings = platefit_continfit(logwl[ok], restwl[ok], flux[ok], err[ok], settings, debug=debug)
if debug:
    plot_bestmodel(logwl[ok], restwl[ok], flux[ok], err[ok], settings)

plt.figure()

plt.plot(restwl, flux, 'k-')  #  spectrum
plt.plot(restwl, best_continuum, 'g-')  # best fit
# plt.plot(restwl, flux - best_continuum, '-', color='y', lw=0.5)  # residuals

plt.show()
del restwl, best_continuum
