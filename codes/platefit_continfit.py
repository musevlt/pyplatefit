#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import sys
import numpy as np

import matplotlib.pyplot as plt

import model_fit_nnls
model_fit_nnls = reload(model_fit_nnls)
from model_fit_nnls import model_fit_nnls



def platefit_continfit(logwl, restwl, flux, err, settings, debug=None):
    """
    Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl)
    Translation of "fiber_continfit.pro" (part of the IDL PLATEFIT - contact: jarle@strw.leidenuniv.nl)

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
            sys.exit('WARNING: the fitting failed')


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


    # Read the stored values into the settings
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


    # --------------------------------------- plot the best model -- debug ---------------------------------------------
    #
    if debug is True:
        print('best model chi squared value: ', best_modelChi)
        print('all model chi squared values = ', model_chi)
        print('best E(B-V) = ', best_contCoefs[0])

        print('best model coefs = ', np.stack((best_contCoefs[1:], settings['ssp_ages']/1.0E6), axis=-1))
        print('best stellar Z:', best_szval)

        perr = err
        pi = np.array(np.where(err == 0)).squeeze()

        if np.size(pi) > 0: perr[pi] = np.NaN

        mask = flux - best_continuum
        mask[np.array(ok_fit).squeeze()] = np.NaN

        plt.figure()
        plt.plot(restwl, mask, 'm-', lw=2.0)  # Plot the mask first, so it is underneath the rest

        plt.plot(restwl, flux, 'k-')  #  spectrum
        plt.plot(restwl, best_continuum, 'g-')  # best fit
        # plt.plot(restwl, flux - best_continuum, '-', color='y', lw=0.5)  # residuals

        nmodels = np.shape(best_modellib)[1]


        # plt.plot(restwl, perr, 'b-', lw=1)
        # plt.plot(restwl, -perr, 'b-', lw=1)
        plt.show()



    return best_continuum, settings
