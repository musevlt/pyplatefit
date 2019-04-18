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
logger.setLevel(logging.INFO)

logger.info('pyplatefit version %s', __version__)
debug = True


#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref24348.fits'
#z= 0.41909


#data_dir = 'PLATEFIT_testdata/'
#name = 'udf_udf10_00296.fits'

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/orig_specs/ref00216.fits'
#z = 0.99738

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref09667.fits'
#z = 1.55051
# emiline fit succeed only (dz=0)

name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/orig_specs/ref00142.fits'
z = 3.749

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref08994.fits'
#z = 3.32571
vdisp = 80.0

# emiline fit succeed only if dz is not used

sp = Spectrum(name)

pl = Platefit()
res_cont = pl.fit_cont(sp, z, vdisp)
#pl.info_cont(res_cont)
#res_line = pl.fit_lines(res_cont['line_spec'], z, lines=['OII3727','OII3729'])
res_line = pl.fit_lines(res_cont['line_spec'], z)
pl.info_lines(res_line)


fig,ax = plt.subplots(1,2)
pl.plot(ax, {**res_line,**res_cont})
plt.show()

#pl.plot(ax, res_line)
#plt.show()




print('end')
