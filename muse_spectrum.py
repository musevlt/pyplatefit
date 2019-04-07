#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import sys

import numpy as np
from mpdaf.sdetect import Source

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
name = 'udf_udf10_00010.fits'
# name = 'udf_mosaic_01011_newmask.fits.gz'
src = Source.from_file(data_dir+name)

sp = src.spectra[src.REFSPEC]
vdisp = 80.0
z = src.z[src.z['Z_DESC']=='MUSE']['Z'][0]
logger.debug('z = %f',z)

cont = pl.contfit(sp, z, vdisp)

fig,ax = plt.subplots(1,1)
sp.plot(ax=ax)
cont.plot(ax=ax)
plt.show()

print('end')
