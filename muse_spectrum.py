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
from pyplatefit.platefit import Platefit, fit_all, fit_one

import logging

logger = logging.getLogger('pyplatefit')
logger.setLevel(logging.INFO)

logger.info('pyplatefit version %s', __version__)
debug = True

vdisp = 80.0

#paths = ['/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref24348.fits', 
         #'/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/orig_specs/ref00216.fits',
         #'/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/orig_specs/ref00142.fits',
         #'/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref09667.fits',
         #'/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/orig_specs/ref00216.fits',
         #'/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref08994.fits'
         #]
#zlist = [0.41909,0.99738,3.749,1.55051,0.99738,3.32571]
#idlist= [24348,216,142,9667,216,8994]
#fromlist = ['HSTPRIOR','ORIGIN','ORIGIN','HSTPRIOR','ORIGIN','HSTPRIOR']
#cat = Table(data=[idlist,fromlist,zlist,paths],
            #names=['ID','FROM','Z','PATH'])

#ztable,ltable = fit_all(cat, njobs=3, emcee=False, comp_bic=True)

#src_tpl = '/Users/rolandbacon/UDF/DR2/sources/raf_sources/source-%05d.fits'
#iden = 6726
#z = 2.9409
#srcname = src_tpl%(iden)
#src = Source.from_file(srcname)
#spec = src.spectra[src.REFSPEC]
#addcols = None
#fit_one(iden, 'HSTPRIOR', z, spec, addcols, src_tpl, emcee=False, comp_bic=False, prefix='MZ1')

name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref24348.fits'
z= 0.41909


#data_dir = 'PLATEFIT_testdata/'
#name = 'udf_udf10_00296.fits'

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/orig_specs/ref00216.fits'
#z = 0.99738

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref09667.fits'
#z = 1.55051


#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/orig_specs/ref00142.fits'
#z = 3.749

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref07407.fits'
#z = 0.844

#name = '/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/DR2/raf_specs/ref08994.fits'
#z = 3.32571



sp = Spectrum(name)

#pl = Platefit(linepars=dict(steps=100))
pl = Platefit()
res = pl.fit(sp, z, emcee=True, vel_uniq_offset=False, eqw=True)
pl.info(res)
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
