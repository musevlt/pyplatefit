#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/pyNNLS")

import numpy as np
from PyAstronomy import pyasl
import matplotlib.pyplot as plt

from nnls_burst_python0 import fit_continuum1


def fit_burst_nnls(flux, wavelength, dflux, ok, settings, settings_nnls):
    """
        Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl)
        Translation of "fit_burst_nnls.pro" (part of the IDL PLATEFIT - contact: jarle@strw.leidenuniv.nl)

    """

    modellibrary = settings['modellib']

    # Weights the flux by the error
    b = np.array(flux[ok]) / np.array(dflux[ok])
    lam = np.array(wavelength[ok]).squeeze()

    dims = np.shape(modellibrary)
    n = dims[1]
    m = np.size(ok)

    # Various other input/output parameters
    params = np.zeros(n+1)  # N models + EBV. Where params[0] will be EBV
    nparams = n + 1

    # Weight the model library by the errors too
    a = np.zeros((m, n))
    for i in range(n):
        temp = np.array(modellibrary[ok, i]) / np.array(dflux[ok])
        a[:, i] = np.array(temp).squeeze()

    # Adjust the input parameters to match the shape and precision
    a = np.array(a).transpose().reshape(1, np.size(a))
    a = np.array(a).squeeze()
    nxm = m
    nmodel = n
    lam = np.array(lam).squeeze()
    flux = b
    params = params
    nparams = nparams
    nm = np.size(a)
    mean = np.zeros(1)  # one of the inputs/outputs of fortran. NOTE the way the variable is declared
    sigma = np.zeros(1)  # one of the inputs/outputs of fortran


    # ----------------------------- Call the external fortran routine --------------------------------------------------
    # fit_continuum1test(np.long(nparams), np.int(nxm), np.long(nmodel), np.long(nm), np.double(lam), np.double(flux),
    #                    np.double(a), np.double(params), np.double(mean), np.double(sigma))
    # fit_continuum1test(lam, flux, params, nxm, nparams)

    # print(nxm, nm, nparams, nmodel)
    # lambda, flux, models, parameters, mean, sigma, nmodels, ok, size(a), nparams
    fit_continuum1(lam, np.double(flux), np.double(a), np.double(params),
                   mean, sigma, nmodel, nxm, nm, nparams)

    # print('parameters=', params)
    # print(mean, sigma)

    # Write the results in to the 'settings_nnls' structure
    settings_nnls['params'] = params
    settings_nnls['mean'] = mean
    settings_nnls['sigma'] = sigma

    return settings_nnls