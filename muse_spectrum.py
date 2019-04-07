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

name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/orig_specs/ref00216.fits'
sp = Spectrum(name)
z = 0.99738

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref54891.fits'
#sp = Spectrum(name)
#z = 1.51052

vdisp = 80.0

logger.debug('z = %f',z)

cont = pl.contfit(sp, z, vdisp)

fig,ax = plt.subplots(1,1)
sp.plot(ax=ax)
cont.plot(ax=ax)
plt.show()

print('end')
