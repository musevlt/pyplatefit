#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/ppxf_versions/ppxf")
import ppxf_util

import numpy as np
from PyAstronomy import pyasl

# The functions provided in these packages can also be used for convolving Gaussian with templates.
# from astropy.convolution import convolve, Gaussian1DKernel


def resample_model(logwl, z, vdisp, modelsz, settings):
    """
        Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl)
        Translation of "resample_model.pro" (part of the IDL PLATEFIT - contact: jarle@strw.leidenuniv.nl)

    """
    nmodels = np.shape(settings['burst_lib'])[1]

    # ------------------------ redshift model wave array, and put in vaccum --------------------------------------------
    obs_burst_wl = settings['burst_wl']
    obs_burst_wl = obs_burst_wl * (1.0 + z)
    obs_burst_wl = pyasl.airtovac(obs_burst_wl)

    burst_lib = settings['burst_lib']

    # de-redshift data wave array, and put in air only for comparison to burst_wl
    npix = np.size(logwl)
    restwl = 10.0**logwl

    # Convert wavelength from vacuum to air
    restwl = pyasl.vactoair(restwl)  # The wavelengths in the linepar are in air
    restwl = restwl / (1.0 + z)

    # Convert wavelength in air to wavelength
    # in vacuum
    # restwl = pyasl.airtovac(restwl)


    # -------------------- Interpolate models to match data & convolve to velocity dispersion --------------------------
    # This is the resolution of the templates
    data_disp = settings['model_dispersion']

    cspeed = 2.99792E5
    loglamtov = cspeed * np.log(10.0)

    dw = logwl[1] - logwl[0]

    # Figure out convolution 'sigma' in units of pixels, being sure to deconvolve the template resolution first
    if vdisp <= data_disp:
        sigma_pix = 50.0 / loglamtov / dw
        print('WARNING: vdisp < the dispersion of the templates. sigma_pix is set to 50.0km/s')
    else:
        vdisp_add = np.sqrt(vdisp**2 - data_disp**2)  # Deconvolve template resolution
        sigma_pix = vdisp_add / loglamtov / dw


    # Interpolate reshifted models to the same wavelength grid as the data (in log-lambda) and then convolve to the
    # data velocity dispersion
    custom_lib = np.zeros((npix, nmodels))
    temp_wl = settings['burst_wl']

    # Another way of doing the convolution (using scipy.convolve). But I am sticking with the ppxf version.
    # gauss_kernel = Gaussian1DKernel(sigma_pix)

    for i in range(nmodels):
        burst = np.interp(logwl, np.log10(obs_burst_wl), burst_lib[:, i, modelsz])
        # burst = np.interp(logwl, np.log10(obs_burst_wl), burst_lib[:, i, modelsz])

        # Smooth the template with a Gaussian filter.
        custom_lib[0:, i] = ppxf_util.gaussian_filter1d(burst, sigma_pix)
        # custom_lib[0:, i] = sp.ndimage.filters.gaussian_filter1d(burst, sigma_pix, order=0)
        # custom_lib[0:, i] = convolve(burst, gauss_kernel)


    # -------------------------- set regions outside of the mode to zero -----------------------------------------------
    outside_model = np.where((restwl <= np.min(temp_wl)) | (restwl >= np.max(temp_wl)))
    if np.size(outside_model) > 0:
        custom_lib[outside_model, :] = 0.0


    # -------------------------- load in to the settings ---------------------------------------------------------------
    settings['modellib'] = custom_lib
    settings['burst_wllib'] = temp_wl


    return settings
