#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import sys
import numpy as np

import matplotlib.pyplot as plt

from .model_fit_nnls import model_fit_nnls

import logging
logger = logging.getLogger('pyplatefit')


def platefit_continfit(logwl, restwl, flux, err, settings):
    """
    Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl)
    Translation of "fiber_continfit.pro" (part of the IDL PLATEFIT - contact: jarle@strw.leidenuniv.nl)

    Jan 18, 2019, Madusha:
    In the case NNLS fitting of the continuum is failed, settings returns best_sz = -99.0

    """

    nsz = np.size(settings['szval'])

    npix = np.size(flux)
    continuum = np.zeros((npix, nsz))
    model_chi = np.zeros(nsz)
    model_contcoeff = np.zeros((np.shape(settings['burst_lib'])[1], nsz))
    model_library = np.zeros((npix, np.shape(settings['burst_lib'])[1], nsz))

    ebv = np.zeros_like(model_chi)

    best_continuum = np.zeros(npix) - 99.0
    best_modelChi = 99999.99
    best_modellib = np.zeros_like(model_library) - 99.0
    best_szval = -99.0
    best_ebv = -99.0

    redshift = np.array(settings['z'])
    flux_temp = np.array(flux[:])


    for isz in range(nsz):
        # Notice that the definition of chi squared is that returned by NNLS
        continuum[:, isz], settings, settings_nnls = model_fit_nnls(logwl, flux, err, redshift, settings['vdisp'], isz,
                                                     settings, firstcall=True, debug=False)

        contcoefs = np.array(settings_nnls['params'][:])
        mean_chi2 = np.array(settings_nnls['mean'][:])

        if contcoefs[0] < -90:
            settings['best_sz'] = -99.0
            settings['best_ebv'] = -99.0
            settings['best_modelChi'] = 99999.99

            return best_continuum, settings


        # Store the best-fit model results
        if mean_chi2 < best_modelChi:
            best_modelChi = mean_chi2
            best_modellib = np.array(settings_nnls['modellib'][:])
            ok_fit = np.array(settings_nnls['ok_fit'][:])
            best_szval = np.array(settings['szval'][isz])
            best_ebv = contcoefs[0]

            best_contCoefs = contcoefs
            best_continuum[:] = continuum[:, isz]

        # Store the model fit results correctly
        ebv[isz] = contcoefs[0]
        model_contcoeff[:,isz] = contcoefs[1:]
        model_chi[isz] = mean_chi2
        model_library[:,:,isz] = np.array(settings_nnls['modellib'][:])

        del contcoefs, mean_chi2, settings_nnls

    del flux
    flux = np.array(flux_temp[:])

    idxbest_sz = np.argmin(np.abs(np.array(settings['szval']) - best_szval))


    
    # save the fitted values into the settings
    settings['best_spectrum'] = best_continuum
    settings['best_sz'] = best_szval
    settings['best_ebv'] = best_ebv
    settings['best_modelChi'] = best_modelChi
    settings['best_ages'] = settings['ssp_ages'][np.array(np.where(best_contCoefs[1:]>0)).squeeze()]
    settings['best_weights'] = best_contCoefs[1:][np.array(np.where(best_contCoefs[1:]>0)).squeeze()]

    # The best spectrum/ages/weight parameters in a given gas- and stellar-metallicity bin
    settings['model_spectrum'] = continuum
    settings['model_library'] = model_library
    settings['model_ebv'] = ebv
    settings['model_weights'] = model_contcoeff
    settings['model_chi'] = model_chi
    settings['model_ebv'] = ebv
    
    settings['ok_fit'] = ok_fit
        
    
    logger.debug('best model chi squared value: %s', best_modelChi)
    logger.debug('all model chi squared values = %s', model_chi)
    logger.debug('best E(B-V) = %s', best_contCoefs[0])

    logger.debug('best model coefs = %s', np.stack((best_contCoefs[1:], settings['ssp_ages']/1.0E6), axis=-1))
    logger.debug('best stellar Z: %s', best_szval)



    return best_continuum, settings


def plot_bestmodel(logwl, restwl, flux, err, settings):
    """ plot the best model """

    # --------------------------------------- plot the best model -- debug ---------------------------------------------
    #


    perr = err
    pi = np.array(np.where(err == 0)).squeeze()

    if np.size(pi) > 0: perr[pi] = np.NaN

    best_continuum = settings['best_spectrum']
    ok_fit = settings['ok_fit']
    mask = flux - best_continuum
    mask[np.array(ok_fit).squeeze()] = np.NaN

    plt.figure()
    plt.plot(restwl, mask, 'm-', lw=2.0)  # Plot the mask first, so it is underneath the rest

    plt.plot(restwl, flux, 'k-')  #  spectrum
    plt.plot(restwl, best_continuum, 'g-')  # best fit

    plt.show()
