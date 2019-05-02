import numpy as np
import os
import sys


from logging import getLogger


from .cont_fitting import Contfit
from .line_fitting import Linefit, plotline
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
            eqw=True):
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
    vel_uniq_offset: boolean
       if True, a unique velocity offset is fitted for all lines except lyman-alpha
       default: False
    lsf: boolean
       if True, use LSF model to take into account the instrumental LSF
       default: True
    eqw: boolean
       if True compute equivalent widths
       
    Return
    ------
    result dict
    result['linetable']: astropy line table
    result['ztable']; astropy z table
    result['spec']: MPDAF original spectrum
    result['cont']: MPDAF spectrum, fitted continuum in observed frame
    result['line']: MPDAF spectrum, continnum removed spectrum in observed frame
    result['linefit']: MPDAF spectrum, fitted emission lines in observed frame
    result['fit']: MPDAF spectrum, fitted line+continuum in observed frame
    result['res_cont']: return dictionary from fit_cont
    result['res_line']: returned ResultObject from fit_lines
        """
        res_cont = self.fit_cont(spec, z, vdisp)
        res_line = self.fit_lines(res_cont['line_spec'], z, major_lines=major_lines, lines=lines, 
                                  emcee=emcee, use_line_ratios=use_line_ratios, lsf=lsf,
                                  vel_uniq_offset=vel_uniq_offset)
        
        if eqw:
            self.eqw.comp_eqw(spec, res_cont['line_spec'], z, res_line.linetable)
        

        return dict(linetable=res_line.linetable, ztable=res_line.ztable, cont=res_cont['cont_spec'], line=res_cont['line_spec'], 
                        linefit=res_line.spec_fit, fit=res_line.spec_fit+res_cont['cont_spec'],
                        spec=spec, res_cont=res_cont, res_line=res_line)
                      
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
    Return
    ------
    result: dict
    result['table_spec'] astropy table with the following columns
       - RESTWL: restframe vacuum wavelength
       - FLUX: data value in restframe
       - ERR: stddev of data value
       - CONTFIT: continuum fit (restframe)
       - CONTRESID: smoothed continuum residual (restframe)
       - CONT: continuum fit + residual
       - LINE: continuum subtracted 
       - AIRWL: observed wavelength in air
    result['cont_spec']: MPDAF spectrum continuum (fit + smooth residual) in observed frame
    result['cont_fit']: MPDAF spectrum continuum (fit only) in observed frame
    result['line_spec']: MPDAF continuum subtracted spectrum in observed frame
    result['success']: True or False
    result['z']: Metallicity
    result['ebv']: E(B-V)
    result['chi2']: Chi2 fit
    result['ages']: fitted ages
    result['weights']: used weights
    
    
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
    vel_uniq_offset: boolean
       if True, a unique velocity offset is fitted for all lines except lyman-alpha
       default: False
    lsf: boolean
       if True, use LSF model to take into account the instrumental LSF
       default: True
        """
        return self.line.fit(line, z, major_lines=major_lines, lines=lines, emcee=emcee, 
                             use_line_ratios=use_line_ratios, 
                             lsf=lsf, vel_uniq_offset=vel_uniq_offset)        
        
    def info_cont(self, res):
        """
        print some info
        """
        self.cont.info(res)
        
    def info_lines(self, res, full_output=False):
        """
        print some info
        """
        self.line.info(res, full_output=full_output)
        
    def info(self, res, full_output=False):
        self.logger.info('++++ Continuum fit info')
        self.cont.info(res['res_cont'])
        self.logger.info('++++ Line fit info')
        self.line.info(res['res_line'], full_output=full_output)
        
        
    def eqw(self, lines_table, spec, smooth_cont, window=50):
        self.eqw.compute_eqw(lines_table, spec, smooth_cont, window=window)
        
    def plot_cont(self, ax, res):
        self.cont.plot(ax, res)
        
    def plot_lines(self, ax, res, start=False, iden=True, minsnr=0, line=None,
                   margin=5, dplot={'dl':2.0, 'y':0.95, 'size':10}):
        self.line.plot(ax, res, start=start, iden=iden, minsnr=minsnr,
                       line=line, margin=margin, dplot=dplot) 
        
    def plot(self, ax, res, mode=2, start=False, iden=True, minsnr=0, line=None,
                   margin=5, dplot={'dl':2.0, 'y':0.95, 'size':10}):
        """
        plot results fo fit
        
        Parameters
        ----------
        ax : axes
           matplotlib ax
        res: dictionary
           result as given by fit
        mode: integer
           plot mode: 0=continuum, 1=line, 2=cont+line
           default: 2
        start: boolean
           use for plot mode = 1, display start fitting value
           default False
        iden: boolean
           if True, display line names (mode=1,2 only)
        minsnr: float
           minimum SNR to display the line name (mode=1,2 only)
           default: 0
        line: string
           name of the line where to center the plot (eg HALPHA)
           defaut None, display the full spectrum
        margin: integer
           number of margin pixel to add for the zoom window
           default: 5
        dplot: dictionary
           dl: offset in A to display the line name [default 2]
           y: location of the line name in y (relative) [default 0.95]
           size: font size for label display [default 10]
        """
        if mode == 0:
            self.cont.plot(ax, res['res_cont'])
        elif mode == 1:
            self.line.plot(ax, res['res_line'], start=start, iden=iden, minsnr=minsnr,
                           line=line, margin=margin, dplot=dplot)  
        elif mode == 2:
            plotline(ax, res['spec'], res['fit'], res['cont'], None, res['linetable'],
                     iden=iden, minsnr=minsnr, line=line, margin=margin, dplot=dplot)            
        else:
            self.logger.error('unknown plot mode (0=cont,1=line,2=full)')
             
            
        
    
            
        
            
        