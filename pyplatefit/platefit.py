import numpy as np
import os
import sys

from astropy.convolution import Box1DKernel, convolve
#from astropy.io import fits
from logging import getLogger
#from mpdaf.obj import airtovac, vactoair
from astropy.table import MaskedColumn

from .cont_fitting import Contfit
from .line_fitting import Linefit


class Platefit:
    """
    This class is a poorman version of Platefit
    """

    def __init__(self, contpars={}, linepars={}):
        self.logger = getLogger(__name__)
        self.cont = Contfit(**contpars)
        self.line = Linefit(**linepars)

    def fit(self, spec, z, vdisp=80, major_lines=False, lines=None, emcee=False, 
            use_line_ratios=True, vel_uniq_offset=False, lsf=True,
            eqw=True, full_output=False):
        """
    Perform continuum and emission lines fit on a spectrum
    
    Parameters
    ----------
    line : mpdaf.obj.Spectrum
       continuum subtracted spectrum
    z : float
       reshift 
    vdisp : float
       velocity dispersion in km/s [default 80 km/s]
    major_lines : boolean
       if true, use only major lines as defined in MPDAF line list
       default False
    lines: list
       list of MPDAF lines to use in the fit
       default None
    emcee: boolean
       if True perform a second fit using EMCEE to derive improved errors (note cpu intensive)
       default False
    use_line_ratios: boolean
       if True, use constrain line ratios in fit
       default True 
    eqw: boolean
       if True compute equivalent widths
    full_output: boolean
       if True, return two objects, res_cont and res_line with the full info
       if False, return only a dictionary  with the line table ['table'], the fitted continuum spectrum ['cont'], 
       the continuum subtracted spectrum ['line'], emission lines fit ['linefit'] and complete fit ['fit']
        """
        res_cont = self.fit_cont(spec, z, vdisp)
        res_line = self.fit_lines(res_cont['line_spec'], z, major_lines=major_lines, lines=lines, 
                                  emcee=emcee, use_line_ratios=use_line_ratios, lsf=lsf,
                                  vel_uniq_offset=vel_uniq_offset)
        
        if eqw:
            cont = self.smooth_cont(spec, res_line.spec_fit)
            self.eqw(res_line.linetable, cont)
        
        if full_output:
            return res_cont,res_line
        else:
            return dict(table=res_line.linetable, cont=res_cont['cont_spec'], line=res_cont['line_spec'], 
                        linefit=res_line.spec_fit, fit=res_line.spec_fit+res_cont['cont_spec'],
                        smoothcont=cont)
                      
    def fit_cont(self, spec, z, vdisp):
        """
    Perform continuum lines fit on a spectrum 
    
    Parameters
    ----------
    line : mpdaf.obj.Spectrum
       continuum subtracted spectrum
    z : float
       reshift 
    vdisp : float
       velocity dispersion in km/s 
        """
        return self.cont.fit(spec, z, vdisp)
    
    def fit_lines(self, line, z, major_lines=False, lines=None, emcee=False, 
                  use_line_ratios=True, vel_uniq_offset=False, lsf=True):
        """  
    Perform emission lines fit on a continuum subtracted spectrum 
    
    Parameters
    ----------
    line : mpdaf.obj.Spectrum
       continuum subtracted spectrum
    z : float
       reshift 
    major_lines : boolean
       if true, use only major lines as defined in MPDAF line list
       default False
    lines: list
       list of MPDAF lines to use in the fit
       default None
    emcee: boolean
       if True perform a second fit using EMCEE to derive improved errors (note cpu intensive)
       default False
    use_line_ratios: boolean
       if True, use constrain line ratios in fit
       default True

        """
        return self.line.fit(line, z, major_lines=major_lines, lines=lines, emcee=emcee, 
                             use_line_ratios=use_line_ratios, 
                             lsf=lsf, vel_uniq_offset=vel_uniq_offset)        
        
    def info_cont(self, res):
        """
        print some info
        """
        self.cont.info(res)
        
    def info_lines(self, res):
        """
        print some info
        """
        self.line.info(res)    
           
            
    def plot(self, ax, res):
        """
        plot results
        """
        axc = ax[0] if type(ax) in [list, np.ndarray] else ax          
        if 'spec' in res:
            res['spec'].plot(ax=axc,  label='Spec')
        if 'cont_spec' in res:
            res['cont_spec'].plot(ax=axc, alpha=0.8, label='Cont Fit')
        axc.set_title('Cont Fit')
        axc.legend()
        if 'line_spec' in res:
            axc = ax[1] if type(ax) in [list, np.ndarray] else ax
            res['line_spec'].plot(ax=axc, label='Line')
            axc.set_title('Line')
            axc.legend()
        if 'line_fit' in res: 
            data_kws = dict(markersize=2)
            res['lmfit'].plot_fit(ax=axc, data_kws=data_kws, show_init=True)
            for row in res['table']:
                axc.axvline(row['LBDA_EXP'], color='r', alpha=0.2) 
                axc.axvline(row['LBDA'], color='k', alpha=0.2) 
                #y1,y2 = axc.get_ylim()
                #axc.text(row['LBDA']+5, y2-0.1*(y2-y1), row['LINE'], color='k', fontsize=8)                   
            axc.set_title('Line Fit')    
            
        
    
    def smooth_cont(self, spec, linefit, kernels=(100,30)):
        """ perform continuum estimation 
        
        Parameters
        ----------
        spec : mpdaf.obj.Spectrum
           raw spectrum
        linefit: mpdaf.obj.Spectrum
           fitted emission lines
        kernels: tuple
           (k1,k2), k1 is the kernel size of the median filter, k2 of the box filter
           default: (100,30)
           
        Return:
        cont : mpdaf.obj.Spectrum
           continuum spectrum
        """
        
        spcont = spec - linefit 
        sm_cont = spcont.median_filter(kernel_size=kernels[0])
        kernel = Box1DKernel(kernels[1])
        sm_cont.data = convolve(sm_cont.data, kernel)
        
        return sm_cont
        
    def eqw(self, lines_table, cont_spec):
        """
        compute equivalent widths, add computed values in lines table
        
        Parameters
        ----------
        lines_table: astropy.table.Table
        
        cont_spec : mpdaf.obj.Spectrum
           continuum spectrum
           
        """
        for name in ['EQW', 'EQW_ERR', 'CONT_OBS', 'CONT', 'CONT_ERR']:
            if name not in lines_table.colnames:
                lines_table.add_column(MaskedColumn(name=name, dtype=np.float, length=len(lines_table), mask=True))
                lines_table[name].format = '.2f'
        
        for line in lines_table:
            name = line['LINE']
            lbda = line['LBDA_OBS']
            fwhm = line['FWHM_OBS']
            z = line['Z']
            # compute continuum flux average over line +/- fwhm 
            sp = cont_spec.subspec(lmin=lbda-fwhm,lmax=lbda+fwhm)
            spmean = np.median(sp.data)
            line['CONT_OBS'] = spmean
            # convert to restframe
            spmean = spmean*(1+z)
            line['CONT'] = spmean
            # compute continuum error
            spmean_err = np.std(sp.data)
            line['CONT_ERR'] = spmean_err
            # compute EQW in rest frame
            eqw = line['FLUX']/spmean
            eqw_err = line['FLUX_ERR']/spmean + line['FLUX']*spmean_err/spmean**2
            if line['FLUX'] > 0:
                line['EQW'] = -eqw
            else:
                line['EQW'] = eqw
            line['EQW_ERR'] = eqw_err
            
        
            
        