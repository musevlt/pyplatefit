import numpy as np
import os
import sys


from logging import getLogger


from .cont_fitting import Contfit
from .line_fitting import Linefit
from .eqw import EquivalentWidth


class Platefit:
    """
    This class is a poorman version of Platefit
    """

    def __init__(self, contpars={}, linepars={}):
        self.logger = getLogger(__name__)
        self.cont = Contfit(**contpars)
        self.line = Linefit(**linepars)
        self.eqw = EquivalentWidth()

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
            smcont = self.eqw.smooth_cont(spec, res_line.spec_fit)
            self.eqw.compute_eqw(res_line.linetable, spec, smcont)
        
        if full_output:
            return res_cont,res_line
        else:
            return dict(table=res_line.linetable, cont=res_cont['cont_spec'], line=res_cont['line_spec'], 
                        linefit=res_line.spec_fit, fit=res_line.spec_fit+res_cont['cont_spec'],
                        smoothcont=smcont)
                      
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
    def eqw(self, lines_table, spec, smooth_cont, window=50):
        self.eqw.compute_eqw(lines_table, spec, smooth_cont, window=window)
           
            
    #def plot(self, ax, res):
        #"""
        #plot results
        #"""
        #axc = ax[0] if type(ax) in [list, np.ndarray] else ax          
        #if 'spec' in res:
            #res['spec'].plot(ax=axc,  label='Spec')
        #if 'cont_spec' in res:
            #res['cont_spec'].plot(ax=axc, alpha=0.8, label='Cont Fit')
        #axc.set_title('Cont Fit')
        #axc.legend()
        #if 'line_spec' in res:
            #axc = ax[1] if type(ax) in [list, np.ndarray] else ax
            #res['line_spec'].plot(ax=axc, label='Line')
            #axc.set_title('Line')
            #axc.legend()
        #if 'line_fit' in res: 
            #data_kws = dict(markersize=2)
            #res['lmfit'].plot_fit(ax=axc, data_kws=data_kws, show_init=True)
            #for row in res['table']:
                #axc.axvline(row['LBDA_EXP'], color='r', alpha=0.2) 
                #axc.axvline(row['LBDA'], color='k', alpha=0.2) 
                ##y1,y2 = axc.get_ylim()
                ##axc.text(row['LBDA']+5, y2-0.1*(y2-y1), row['LINE'], color='k', fontsize=8)                   
            #axc.set_title('Line Fit')    
            
        
    
            
        
            
        