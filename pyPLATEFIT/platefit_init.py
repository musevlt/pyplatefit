#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import os
import sys
import numpy as np
from astropy.io import fits

CURDIR = os.path.dirname(os.path.abspath(__file__))


def platefit_init():
    # ---------------------------------------------- GENERAL SETTINGS --------------------------------------------------
    settings = {
        # directories
        'pipelinedir': CURDIR,
        'twoddir': '',
        'oneddir': '',
        'dustdir': '',
        'platefitdir': '',

        # line fit information
        'linepars': {},
        'indexpars': {},
        'obspars': {},
        'obstags': {},

        # burst model files
        'burst_model_file': 'BC03/bc_models_subset_cb08_miles_v1_bursts.fit',

        'available_z': [0.0001, 0.0004, 0.001,  0.004,  0.008,  0.017,  0.04,   0.07],
        'use_z': [0.0001, 0.0004, 0.001,  0.004,  0.008,  0.017,  0.04,   0.07 ],
        'linelist': 'etc/muse_antennae_linelist_subset.txt',

        # burst models information to be recorded
        'burst_lib': [],
        'burst_wl': [],
        'wavelength_range_for_fit': [2350., 4550.],
        'gzval': [],
        'szval': [],
        'gztags': [],
        'sztags': [],
        'ssp_ages': [],
        'ssp_norms': [],
        'model_dispersion': [],
    }

    cspeed = 2.99752e5
    zsolar = 0.02

    # ------------------------------------ Load instantaneous burst model files ----------------------------------------
    # Read back metallicities from FITS headers (assume these are  correct!)
    n_met_all = 8
    zmod = np.zeros(n_met_all)

    hdulist = fits.open(settings['pipelinedir'] + settings['burst_model_file'],
                        mode='denywrite', memmap=True, do_not_scale_image_data=True)

    # Read metallicities from header fields
    for iz in range(n_met_all):
        zmod[iz] = hdulist[iz].header['Z']

    # Get header information for wavelength arrays
    n_pixmodel = hdulist[0].header['NAXIS1']
    n_burst = hdulist[0].header['NAXIS2']
    wl0 = hdulist[0].header['CRVAL1']
    dw = hdulist[0].header['CD1_1']
    burst_wl = np.arange(n_pixmodel) * dw + wl0

    # Create the model arrays using only the metallicities specified by the user in the settings
    n_met = len(settings['use_z'])

    burst_lib = np.zeros((n_pixmodel, n_burst, n_met))
    ssp_norms = np.zeros((n_burst, n_met))

    if 'burst' in settings['burst_model_file']:
     all_norms = hdulist[8].data[:]

    for imod in range(n_met):
        indx = np.array(np.where(zmod == settings['use_z'][imod])).squeeze()

        try:
            tmp = hdulist[np.int(indx)].data[:]
        except ValueError:
            sys.exit('ABORT -- requested model metallicity not found!')

        burst_lib[:, :, imod] = np.array(tmp).transpose()

        if 'burst' in settings['burst_model_file']:
            ssp_norms[:, imod] = all_norms[:, imod]

    ssp_ages = hdulist[n_met_all + 1].data[:]


    # model dispersion
    model_dispersion = 75.0
    settings['model_dispersion'] = model_dispersion

    settings['burst_lib'] = burst_lib
    settings['ssp_norms'] = ssp_norms
    settings['burst_wl'] = burst_wl
    settings['ssp_ages'] = ssp_ages
    settings['szval'] = settings['use_z']

    del hdulist

    return settings


