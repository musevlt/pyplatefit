#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
# __all__ = ["format_output"]

import sys

import numpy as np
from mpdaf.sdetect import Source
from mpdaf.obj.spectrum import Spectrum
from astropy.table import Table

import matplotlib.pyplot as plt
from pyplatefit import __version__
from pyplatefit.platefit import Platefit, fit_spec, plot_fit

import logging

logger = logging.getLogger('pyplatefit')
logger.setLevel(logging.DEBUG)

logger.info('pyplatefit version %s', __version__)
debug = True

vdisp = 80.0

name = '/Users/rolandbacon/Dropbox/Soft/python/pyplatefit/tests/test_data/udf10_00605.fits'
zinit = 3.43774

#name = '/Users/rolandbacon/Dropbox/Soft/python/pyplatefit/tests/test_data/udf10_00723.fits'
#z = 3.18817

#name = '/Users/rolandbacon/Dropbox/Soft/python/pyplatefit/tests/test_data/udf10_00056.fits'
#z = 1.30604

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/UDF10/Final/DR2_udf10_000043.fits'
#z = 1.09989

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/UDF10/Final/tests/sptest.fits'
#zinit = 3.17140
#z = 3.17136

#z = z + 400/300000.0
#print(z)

#name = '/Users/rolandbacon/Dropbox/MUSE/Anna/spec_LAE_stack_platefit.fits'
#zinit = 0

name = '/Users/rolandbacon/Dropbox/Soft/python/pyplatefit/tests/test_data/udf10_00106.fits'
zinit = 3.27554

sp = Spectrum(name)
res = fit_spec(sp, zinit, lines=['LYALPHA'], emcee=True, dble_lyafit=True, find_lya_vel_offset=False)

#res2 = fit_spec(sp, z)
#                linepars=dict(delta_vel=100, delta_vdisp=50, delta_gamma=None))

fig,ax = plt.subplots(1,1,figsize=(10,5))
plot_fit(ax, res, line='LYALPHA', line_only=True, start=True)
res['ztable'].pprint_all()
print(res['lines'].loc['LYALPHA'])

#ax[0].set_title('LSQ Fit')
#plot_fit(ax[1], res2, line='CIV1548', line_only=True, start=True)
#ax[1].set_title('EMCEE Fit')
#plot_fit(ax, res, filterspec=20)
plt.show()

#res = fit_spec(sp, z, emcee=True, ziter=False, find_lya_vel_offset=True, use_line_ratios=True)
#res = fit_spec(sp, z, emcee=True)
#res = fit_spec(sp, z, emcee=True, linepars=dict(progress=True))




#pl = Platefit()
#res = pl.fit(sp, z)
#pl.fit(sp, z, emcee=False, lines=['LYALPHA'])


#pl.info(res)
#fig,ax = plt.subplots(1,1)
##pl.plot_cont(ax, res)
#pl.plot_lines(ax, res, line=)
#plt.show()

#res = fit_spec(sp, z, emcee=False, comp_bic=True, lines=['OII3727','OII3729'], use_line_ratios=True,
               #linepars=dict(line_ratios=[("OII3727", "OII3729", 0.5, 0.8)])
               #)
#res = fit_spec(sp, z, ziter=False, lines=['OII3727','OII3729'], use_line_ratios=True, 
               #linepars=dict(line_ratios=[("OII3727", "OII3729", 0.5, 0.8)]))


#pl = Platefit(linepars=dict(steps=100))
#res2 = pl.fit(sp, z, emcee=False, vel_uniq_offset=False, eqw=True, trimm_spec=True)
#pl.info(res2)

#if 'OII3727' in res['linetable']['LINE']:
    #lines = ['OII3727','OII3729']
#res2 = pl.fit_lines(res['line'], z, lines=lines, trimm_spec=True)
#pl.info_lines(res2, full_output=False)
#res3 = pl.fit_lines(res['line'], z, lines=lines, trimm_spec=False)
#pl.info_lines(res3, full_output=False)


#fig,ax = plt.subplots(1,1)
#pl.plot(ax, res, line='HALPHA', margin=30)
##pl.plot_lines(ax, res['res_line'], line='HALPHA', margin=30)
#plt.show()




#res_cont = pl.fit_cont(sp, z, vdisp)
#pl.info_cont(res_cont)
#res_line = pl.fit_lines(res_cont['line_spec'], z, lines=['OII3727','OII3729'])
#res_line = pl.fit_lines(res_cont['line_spec'], z, emcee=False, major_lines=True, use_line_ratios=True)
#pl.info_lines(res_line)
#print(res_line.linetable)


#fig,ax = plt.subplots(1,2)
#pl.plot(ax, {**res_line,**res_cont})
#plt.show()

#pl.plot(ax, res_line)
#plt.show()




print('end')
