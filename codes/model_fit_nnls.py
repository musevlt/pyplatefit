#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import sys
import numpy as np
from PyAstronomy import pyasl
import matplotlib.pyplot as plt

import resample_model
resample_model = reload(resample_model)
from resample_model import resample_model

import fit_burst_nnls
fit_burst_nnls = reload(fit_burst_nnls)
from fit_burst_nnls import fit_burst_nnls

import model_combine
model_combine = reload(model_combine)
from model_combine import model_combine


def model_fit_nnls(logwl, flux, err, redshift, vdisp, modelsz, settings, firstcall=None, debug=False):
    """
        Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl)
        Translation of "bc_model_fit_nnls.pro" (part of the IDL PLATEFIT - contact: jarle@strw.leidenuniv.nl)

    """
    nmodels = np.shape(settings['burst_lib'])[1]

    z = settings['z']
    cspeed = 2.99792E5

    # de-redshift the data wave array and put in air
    npix = np.size(logwl)
    restwl = 10.0**logwl
    restwl = pyasl.vactoair(restwl)
    restwl = restwl / (1.0 + settings['z'])

    # Interpolate models to match the data and convolve to velocity dispersion
    if firstcall is True:
        settings = resample_model(logwl, settings['z'], settings['vdisp'], modelsz, settings)

    modellib = settings['modellib']


    # -------------------------------------- Apply masks ---------------------------------------------------------------

    outside_model = np.where((restwl <= settings['wavelength_range_for_fit'][0]) |
                             (restwl >= settings['wavelength_range_for_fit'][1]))

    quality = np.zeros(npix) + 1
    if np.size(outside_model) > 0: quality[outside_model] = 0

    # Grow masks a bit
    bad = np.where((np.isfinite(flux) == False) | (np.isfinite(err) == False) | (err == 0))
    if np.size(bad) > 0:
        for i in range(-2, 2): quality[bad + i] = 0

    ok = np.array(np.where(quality == 1)).squeeze()

    if np.size(bad) > 0:
        for i in [-2, -1, 0, 1, 2]:
            quality[bad + i] = 0

    weight = np.zeros(npix)
    weight[ok] = 1.0/err[ok]**2.0


    em= [3703.86, #             ; He I       1
        3726.03, #              ; [O II]     2
        3728.82, #              ; [O II]     3
        3750.15, #              ; H12        4
        3770.63, #              ; H11        5
        3797.90, #              ; H10        6
        3819.64, #              ; He I       7
        3835.38, #              ; H9         8
        3868.75, #              ; [Ne III]   9
        3889.05, #              ; H8        10
        3970.07, #              ; H-episilon 11
        4101.73, #              ; H-delta   12
        4026.21, #              ; He I      13
        4068.60, #              ; [S II]    14
        4340.46, #              ; H-gamma   15
        4363.21, #              ; [O III]
        4471.50, #              ; He I
        4861.33, #              ; H-beta    18
        4959.91, #              ; [O III]
        5006.84, #              ; [O III]
        5200.26, #              ; [N I]
        5875.67, #              ; He I
        5890.0 , #              ; Na D (abs)
        5896.0 , #              ; Na D (abs)
        6300.30, #              ; [O I]
        6312.40, #              ; [S III]
        6363.78, #              ; [O I]
        6548.04, #              ; [N II]
        6562.82, #              ; H-alpha   29
        6583.41, #              ; [N II]
        6678.15, #              ; He I
        6716.44, #              ; [S II]
        6730.81, #              ; [S II]
        7065.28, #              ; He I
        7135.78, #              ; [Ar III]
        7319.65, #              ; [O II]
        7330.16, #              ; [O II]
        7751.12, #              ; [Ar III]
        5577./(1.0+z)] #         ; night sky line


    # The full emission mask suit, mosltly relevant for the Antennae spectra
    # mask out emission lines, NaD, 5577 sky lines
    """ 
    em = [3703.86,  # He I       1
          3726.03,  # [O II]     2
          3728.82,  # [O II]     3
          3750.15,  # H12        4
          3770.63,  # H11        5
          3797.90,  # H10        6
          3819.64,  # He I       7
          3835.38,  # H9         8
          3868.75,  # [Ne III]   9
          3889.05,  # H8        10
          3970.07,  # H-epsilon 11
          4101.73,  # H-delta   12
          4026.21,  # He I      13
          4068.60,  # [S II]    14
          4340.46,  # H-gamma   15
          4363.21,  # [O III]   16
          4471.50,  # He I      17
          4861.33,  # H-beta    18
          4921.90,  #           19
          4959.91,  # [O III]   20
          5006.84,  # [O III]   21
          5016.60,  # HeI       22
          5200.26,  # [N I]     23
          5754.30,  # [N II]    24
          5875.67,  # He I      25
          5890.0,  # Na D (abs) 26
          5896.0,  # Na D (abs) 27
          6300.30,  # [O I]     28
          6312.40,  # [S III]   29
          6363.78,  # [O I]     30
          6548.04,  # [N II]    31
          6562.82,  # H-alpha   32
          6583.41,  # [N II]    33
          6678.15,  # He I      34
          6716.44,  # [S II]    35
          6730.81,  # [S II]    36
          7065.28,  # He I      37
          7135.78,  # [Ar III]  38
          7281.10,  # [HeI]     39
          7319.65,  # [O II]    40
          7330.16,  # [O II]    41
          7751.12,  # [Ar III]  42
          # 8046.00,
          8188.00,  #           43
          # 8216.60,
          # 8223.60,
          8398.00 / (1. + z),  # night sky line  44
          8392.60,  #           45
          8381.80 / (1. + z),  # night sky line  46
          8413.32,  # Pa19      47
          8437.96,  # Pa18      48
          8446.48,  # OI        49
          8467.26,  # Pa17      50
          8502.49,  # Pa16      51
          8545.38,  # Pa15      52
          8578.70,  # CIII      53
          8598.39,  # Pa14      54
          8665.02,  # Pa13      55
          8750.48,  # Pa12      56
          8862.79,  # Pa11      57
          9014.91,  # Pa10      58
          9068.90,  # SIII      59
          9229.02,  # Pa9       60
          5577. / (1. + z),  # night sky line
          6300. / (1. + z),  # night sky line
          6364. / (1. + z)]  # night sky line
    """

    # make mask width a multiple of the velocity dispersion
    if vdisp < 100.0:
        mask_width = 100.0
    elif vdisp > 500.0:
        mask_width = 500.0
    else:
        mask_width = vdisp

    mask_width = mask_width * 5.0
    mask_width = mask_width + np.zeros(np.size(em))

    for i in range(np.size(em)):
        voff = np.abs(restwl - em[i]) / em[i] * cspeed
        maskout = np.where(voff < mask_width[i])

        if np.size(maskout) > 0: quality[maskout] = 0


    ok = np.array(np.where(quality == 1)).squeeze()
    not_ok = np.array(np.where(quality == 0)).squeeze()

    # ----------------------------------- Call the fitting routine -----------------------------------------------------

    settings_nnls = {}  # Declare the 'settings_nnls' structure which will hold the NNLS outputs
    settings_nnls = fit_burst_nnls(flux, restwl, 1.0/np.sqrt(weight), ok, settings, settings_nnls)

    fitcoefs = settings_nnls['params']
    yfit = model_combine(restwl, fitcoefs, settings, settings_nnls)

    settings_nnls['ok_fit'] = ok
    settings_nnls['not_ok_fit'] = not_ok
    settings_nnls['modellib'] = modellib

    # ------------------------------------------------------------------------------------------------------------------
    # If debugging plot spectrum, best fit, and individual stellar components
    # ------------------------------------------------------------------------------------------------------------------
    if debug is True:
        ymax = np.max(yfit) * 1.1

        plt.figure()
        plt.plot(restwl[ok], flux[ok], 'k-')

        plt.plot(restwl, flux-400., 'y-')
        plt.plot(restwl, yfit, 'r-')

        for i in range(nmodels):
            yi = fitcoefs[i+1] * modellib[:, i] * \
                 np.exp(-fitcoefs[0] * (restwl / 5500.0)**(-0.7))
            plt.plot(restwl, yi, 'b-')

        plt.xlim([4600., 9300.]); plt.ylim([0.0, ymax])
        plt.show()


    return yfit, settings, settings_nnls
