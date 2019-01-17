#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import numpy as np


def model_combine(x, a, settings, settings_nnls, good_data=None, ssp_ages=None, individual=False, correct=None):
    """
        Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl)
        Translation of "bc_model_combine.pro" (part of the IDL PLATEFIT - contact: jarle@strw.leidenuniv.nl)

    """
    #
    # At the moment - CORRECT is for using the same as the Fortran code does,
    # namely no reddening difference between young and old population
    #

    modellib = settings['modellib']
    sz = np.shape(modellib)
    nxlib = sz[0]
    nmodels = sz[1]

    if good_data is None: good_data = np.arange(nxlib)
    ngood = np.size(good_data)

    if ssp_ages is None: ssp_ages = np.zeros(nmodels)

    # ------------------------------------------------------------------------------------------------------------------
    # Create a linear combination of templates
    # Redden using the Charlot & Fall law with the time dependent normalisation
    # F_obs = F_int * exp(-Tau_V * (lambda / 5500 A)^-0.7)
    # ------------------------------------------------------------------------------------------------------------------

    # Old method as indicated in Jarle's code
    # y = np.matmul(modellib[good_data, :], a[1:, :])
    # klam = (x / 5500.0)**(-0.7)
    # e_tau_lam = np.exp(-a[0] * klam)
    # y = y * e_tau_lam

    y = 0.0
    klam = (x / 5500.0)**(-0.7)

    if individual is True:
        individuals = modellib * 0.0


    # ----------------------The original Bug report from 'bc_model_combine.pro'-----------------------------------------
    # Bug: 21/07/2013 - this is in principle fine, but the fitting routine does not treat this age cut properly.
    #                   Thus the scaling of the youngest component is incorrect. This is probably not a big deal
    #                   for the emission line fitting because this incorporates a smooth correction to the fit
    #                   but still it should be fixed.
    # ------------------------------------------------------------------------------------------------------------------

    for i in range(nmodels):
        if ssp_ages[i] < 1.0E7: norm = 1.0
        else: norm = 1.0/3.0

        # CORRECT routine!
        if correct is not None: norm = 1.0

        tmp = modellib[good_data, i] * a[i+1] * np.exp(-a[0] * norm * klam)
        y = y + tmp

        if individual is True:
            individuals[good_data, i] = tmp


    # ------------------------------------------------------------------------------------------------------------------
    # Calculate the dy/da partial derivatives
    # Not correct any longer -- redo in using again
    # The commneted out IDL routine from 'bc_model_combine.pro' is written below
    # ------------------------------------------------------------------------------------------------------------------
    #
    # if n_params() gt 2 then begin
    #     pder = dblarr(ngood, nmodels+1)
    #     pder[*, 0] = -y * klam
    #     for i=0, nmodels - 1 do pder[*,i+1] = modellib[good_data, i] * e_tau_lam
    # endif
    #
    # ------------------------------------------------------------------------------------------------------------------

    # Write to the settings_nnls structure
    if individual is True: settings_nnls['individuals'] = individuals

    return np.array(y)


























