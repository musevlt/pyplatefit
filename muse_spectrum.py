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

#data_dir = 'PLATEFIT_testdata/'
#name = 'udf_udf10_00296.fits'

name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/orig_specs/ref00216.fits'
z = 0.99738
# emiline fit failed in all case

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref09667.fits'
#z = 1.55051
# emiline fit succeed only (dz=0)

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref24348.fits'
#z= 0.41907
# emiline fit succeed only if dz is not used

sp = Spectrum(name)

vdisp = 80.0

logger.debug('z = %f',z)

cont,dz = pl.contfit(sp, z, vdisp)
logger.info('dz=%f',dz)
z = z + dz

#fig,ax = plt.subplots(1,1)
#sp.plot(ax=ax)
#cont.plot(ax=ax)

line = sp - cont
res = line.fit_lines(z, return_lmfit_info=True)


print(f"z: {res['z']:.5f} err: {res['z_err']:.5f} offset: {res['z_off']:.5f}")
print(f"vdisp: {res['vdisp']:.2f} err: {res['vdisp_err']:.5f}")

fig,ax = plt.subplots(1,2)
sp.plot(ax=ax[0])
cont.plot(ax=ax[0])
line.plot(ax=ax[1])
res['bestfit'].plot(ax=ax[1])

plt.show()

print('end')
