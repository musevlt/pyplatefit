#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import sys

import numpy as np
from mpdaf.sdetect import Source
from mpdaf.obj.spectrum import Spectrum

import matplotlib.pyplot as plt
from pyplatefit import __version__
from pyplatefit.platefit import Platefit

import logging

logger = logging.getLogger('pyplatefit')
logger.setLevel(logging.DEBUG)

logger.info('pyplatefit version %s', __version__)
debug = True



pl = Platefit()

data_dir = 'PLATEFIT_testdata/'
name = 'udf_udf10_00296.fits'

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/orig_specs/ref00216.fits'
#z = 0.99738

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref09667.fits'
#z = 1.55051

name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref24348.fits'
z= 0.41907

sp = Spectrum(name)

vdisp = 80.0

logger.debug('z = %f',z)

cont,dz = pl.contfit(sp, z, vdisp)
logger.info('dz=%f',dz)
z = z + dz

line = sp - cont
result_dict, line_table, reslmfit = line.fit_lines(z)
linefit = sp.clone()
linefit.data = np.interp(sp.wave.coord(), reslmfit.wave, reslmfit.best_fit) 

fig,ax = plt.subplots(1,2)
sp.plot(ax=ax[0])
cont.plot(ax=ax[0])
line.plot(ax=ax[1])
linefit.plot(ax=ax[1])

plt.show()

print('end')
