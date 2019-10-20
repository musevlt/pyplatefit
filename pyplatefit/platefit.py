import logging
import os
from astropy.table import vstack, Column, MaskedColumn
from joblib import delayed, Parallel
from mpdaf.obj import Spectrum
from mpdaf.tools import progressbar

from .cont_fitting import Contfit
from .eqw import EquivalentWidth
from .line_fitting import Linefit, plotline


__all__ = ('Platefit', 'fit_spec')


class Platefit:
    """
    This class is a poorman version of Platefit.
    """

    def __init__(self, contpars={}, linepars={}, eqwpars={}):
        """Initialise a Platefit object
        
        Parameters
        ----------
        
        contpars: dictionary
          input parameters to be passed to `Contfit` constructor
          
        linepars: dictionary
          input parameters to be passed to `Linefit` constructor
          
        eqwpars: dictionary
          input parameters to be passed to `EquivalentWidth` constructor
           
        """
        self.logger = logging.getLogger(__name__)
        self.cont = Contfit(**contpars)
        self.line = Linefit(**linepars)
        self.eqw = EquivalentWidth(**eqwpars)

    def fit(self, spec, z, ziter=False, fitcont=True, eqw=True, **kwargs):
        """Perform continuum and emission lines fit on a spectrum

        Parameters
        ----------
        spec : mpdaf.obj.Spectrum
           continuum subtracted spectrum
        z : float
           initial reshift
        ziter : bool
           if True, a first emission line fit is performed to refine the redshift before a new continuum subtraction
           and a complete line fit is performed (to save computation time, eemce option is disactivated for the first fit),
           default false
        fitcont : bool
           Fit and remove continuum prior to line fitting.
        eqw : bool
           Compute Equivalent Width
        **kwargs : keyword arguments
           Additional arguments passed to `Linefit.fit` function.  

        Returns
        -------
        result : dict
        
            - result['lines']: astropy line table (see `fit_lines`)
            - result['ztable']; astropy z table (see `fit_lines`)
            - result['spec']: MPDAF original spectrum
            - result['cont_spec']: MPDAF spectrum, estimated continuum in observed
              frame (cont_fit + smooth residuals)
            - result['cont_fit']: MPDAF spectrum, fitted continuum in observed
              frame             
            - result['line_spec']: MPDAF spectrum, continnum removed spectrum in
              observed frame
            - result['line_fit']: MPDAF spectrum, fitted emission lines in
              observed frame
            - result['line_initfit']: MPDAF spectrum, starting solution for emission line fit in
              observed frame
            - result['spec_fit']: MPDAF spectrum, fitted line+continuum in
              observed frame
            - result['dcont']: return dictionary from fit_cont (see `fit_cont`)
            - result['dline']: returned dictionary from fit_lines (see `fit_lines`)

        """
        
        resfit = {'spec':spec}
        
        if ziter and fitcont:
            self.logger.debug('Performing a first quick fit to refine the input redshift')
            resfit['iter_zinit'] = z
            vdisp = kwargs.pop('vdisp', 80)
            rescont = self.fit_cont(spec, z, vdisp)
            linespec = rescont['line_spec'] 
            # set parameters to speed the fit
            kwargs1 = kwargs.copy()
            kwargs1['emcee'] = False
            kwargs1['fit_all'] = True
            resline = self.fit_lines(linespec, z, **kwargs1)
            ztable = resline['ztable']
            row = ztable[ztable['FAMILY']=='all'][0]               
            vel = row['VEL']
            z = row['Z']
            resfit['iter_z'] = z
            resfit['iter_vel'] = vel      
            self.logger.debug('Computed velocity offset %.1f km/s', vel)
            
        
        if fitcont:
            vdisp = kwargs.pop('vdisp', 80)
            rescont = self.fit_cont(spec, z, vdisp)
            linespec = rescont['line_spec']
            for key in ['cont_spec','cont_fit','line_spec']:
                if key in rescont.keys():
                    resfit[key] = rescont.pop(key)
            resfit['dcont'] = rescont
        else:
            resfit['line_spec'] = spec
            res_cont = {}
            linespec = spec
        
        resline = self.fit_lines(linespec, z, **kwargs)
        
        if eqw and fitcont:
            self.eqw.comp_eqw(spec, linespec, z, resline['lines'])
        
        resfit['lines'] = resline.pop('lines')
        resfit['ztable'] = resline.pop('ztable')
        resfit['line_spec'] = resline.pop('line_spec')
        resfit['line_fit'] = resline.pop('line_fit')
        resfit['line_initfit'] = resline.pop('line_initfit')
        if fitcont:
            resfit['spec_fit'] = resfit['cont_spec'] + resfit['line_fit']
        else:
            resfit['spec_fit'] = resfit['line_fit']
        resfit['dline'] = resline

        return resfit

    def fit_cont(self, spec, z, vdisp):
        """Perform continuum lines fit on a spectrum

        Parameters
        ----------
        line : mpdaf.obj.Spectrum
            Continuum subtracted spectrum
        z : float
            Reshift
        vdisp : float
            Velocity dispersion in km/s

        Returns
        -------
        result : dict
        
            - result['table_spec'] astropy table with the following columns:

              - RESTWL: restframe vacuum wavelength
              - FLUX: data value in restframe
              - ERR: stddev of data value
              - CONTFIT: continuum fit (restframe)
              - CONTRESID: smoothed continuum residual (restframe)
              - CONT: continuum fit + residual
              - LINE: continuum subtracted
              - AIRWL: observed wavelength in air

            - result['cont_spec']: MPDAF spectrum continuum (fit + smooth
              residual) in observed frame
            - result['cont_fit']: MPDAF spectrum continuum (fit only) in
              observed frame
            - result['line_spec']: MPDAF continuum subtracted spectrum in
              observed frame (spec - cont_spec)
            - result['success']: True or False
            - result['z']: Metallicity
            - result['ebv']: E(B-V)
            - result['chi2']: Chi2 fit
            - result['ages']: fitted ages
            - result['weights']: used weights
            
            
        """
        return self.cont.fit(spec, z, vdisp)

    
    def fit_lines(self, line, z, **kwargs):
        """  
        Perform emission lines fit on a continuum subtracted spectrum 
    
        Parameters
        ----------
        line : mpdaf.obj.Spectrum
            continuum subtracted spectrum
        z : float
            reshift 
        kwargs : keyword arguments
           Additional arguments passed to the `fit_lines` function.

       
        Returns
        -------
        res : dictionary
           See `Linefit.fit`
           
        """
        return self.line.fit(line, z, **kwargs)        
        

    def info_cont(self, res):
        """ print a summary of continuum fit results
        
        
        Parameters
        ----------
        res:  
          results from `fit_cont`
        """
        self.cont.info(res)

    def info_lines(self, res, full_output=False):
        """ print a summary of lines fit results
        
        Parameters
        ----------
        res:  
          results from `fit_lines`
        full_output: bool
          if True write more information, default False      
        """
        self.line.info(res, full_output=full_output)

    def info(self, res):
        """ print fitting info form `fit` 
        
        Parameters
        ----------
        res:  
          results from `fit`
        full_output: bool
          if True write more information, default False      
        """
        self.logger.info('++++ Continuum fit info')
        self.cont.info(res['dcont'])
        self.logger.info('++++ Line fit info')
        self.line.info(res)

    def comp_eqw(self, spec, line_spec, z, lines_table):
        """ compute equivalent width for emission lines
        
        Parameters
        ----------
        spec : mpdaf.obj.Spectrum
           raw spectrum 
        line_spec : mpdaf.obj.Spectrum
           continuum subtracted spectrum 
        z : float
           redshift
        lines_table: astropy.table.Table
           input/output lines table to be updated with eqw results
           
        Returns
        -------
        None 
        
        The lines_table will be updated with the following columns:
        
          - EQW : Rest frame equivalent width in A, by convention emission lines have negative eqw
          - EQW_ERR : Standard error in equivalent width in A
        
        """
        self.eqw.comp_eqw(spec, line_spec, z, lines_table)

    def plot_cont(self, ax, res):
        self.cont.plot(ax, res)

    def plot_lines(self, ax, res, start=False, iden=True, minsnr=0, line=None,
                   margin=5, dplot={'dl': 2.0, 'y': 0.95, 'size': 10}):
        self.line.plot(ax, res, start=start, iden=iden, minsnr=minsnr,
                       line=line, margin=margin, dplot=dplot)



def fit_spec(spec, z, fit_all=False, emcee=False, ziter=False, comp_bic=False, fitcont=True, lines=None, 
             major_lines=False, vdisp=80, use_line_ratios=False, find_lya_vel_offset=True,
             lsf=True, eqw=True, trimm_spec=True, contpars={}, linepars={}):
    """ 
    perform platefit cont and line fitting on a spectra
    
    Parameters
    ----------
    spec : MPDAF spectrum or str
      Input spectrum either as a string or a MPDAF spectrum object
    z : float
      redshift (in vacuum)
    fit_all : bool
      If True, fit all lines except Lya together with the same velocity and velocity dispersion (default False)
    emcee : bool
      if True perform a second fit using EMCEE to derive improved errors
      (note cpu intensive), default False.
    ziter : bool
      if True, a first emission line fit is performed to refine the redshift before a new continuum subtraction
      and a complete line fit is performed (to save computation time, eemce option is disactivated for the first fit),
      default false
    comp_bic : bool
      If True compute Bayesian Information Criteria for some lines (default False)
    fitcont : bool
      If True fit and subtract the continuum, otherwise perform only line emission fit (default True)
    lines: list
       list of MPDAF lines to use in the fit (default None). 
    major_lines : bool
       if true, use only major lines as defined in MPDAF line list (default False).
    vdisp : float
       velocity dispersion in km/s (default 80 km/s).
    use_line_ratios : bool
       if True, use constrain line ratios in fit (default False)
    find_lya_vel_offset : bool
       if True, perform an initial search for the lya velocity offset
    lsf : bool
       if True, use LSF model to take into account the instrumental LSF (default True).
    eqw : bool
       if True compute equivalent widths (default True).
    trimm_spec : bool
       if True, trimmed spec around selected emission lines (default True).    
    contpars : dictionary
      Input parameters to pass to `Contfit` (default {})
    linepars : dictionary
      Input parameters to pass to `Linefit` (default {})
  

    Returns
    -------
    result : dict
    
        - result['lines']: astropy line table (see `fit_lines`)
        - result['ztable']; astropy z table (see `fit_lines`)
        - result['spec']: MPDAF original spectrum
        - result['cont_spec']: MPDAF spectrum, estimated continuum in observed
          frame (cont_fit + smooth residuals)
        - result['cont_fit']: MPDAF spectrum, fitted continuum in observed
          frame             
        - result['line_spec']: MPDAF spectrum, continnum removed spectrum in
          observed frame
        - result['line_fit']: MPDAF spectrum, fitted emission lines in
          observed frame
        - result['line_initfit']: MPDAF spectrum, starting solution for emission line fit in
          observed frame
        - result['spec_fit']: MPDAF spectrum, fitted line+continuum in
          observed frame
        - result['dcont']: return dictionary from fit_cont (see `fit_cont`)
        - result['dline']: returned dictionary from fit_lines (see `fit_lines`)

    """
    logger = logging.getLogger(__name__)
    if isinstance(spec, str):
        spec = Spectrum(spec)    
    pl = Platefit(contpars=contpars, linepars=linepars)
    logger.debug('Performing continuum and line fitting')
    res = pl.fit(spec, z, emcee=emcee, fit_all=fit_all, ziter=ziter, fitcont=fitcont,
                 lines=lines, use_line_ratios=use_line_ratios, find_lya_vel_offset=find_lya_vel_offset,    
                 lsf=lsf, eqw=eqw, vdisp=vdisp, trimm_spec=trimm_spec)
    if comp_bic:
        logger.debug('Adding BIC info to ztable for a subset of lines')
        add_bic_to_ztable(res['ztable'], res)
    return res   
        
        
def add_bic_to_ztable(ztab, res):
    """ add BICinfo to ztable """  
    pl = Platefit()
    for name in ['BIC_LYALPHA','BIC_OII','BIC_CIII']:
        ztab.add_column(MaskedColumn(name=name, dtype=float, length=len(ztab), mask=True))
        ztab[name].format = '.2f'
    lines = res['lines']
    # compute bic for individual lines (lya,oii,ciii)
    if 'OII3727' in lines['LINE']:
        linelist = ['OII3727','OII3729']
        z = lines[lines['LINE']=='OII3727']['Z'][0]
        res3 = pl.fit_lines(res['line_spec'], z, emcee=False, lines=linelist)
        if 'forbidden' in ztab['FAMILY']:
            ksel = ztab['FAMILY'] == 'forbidden'
            ztab['BIC_OII'][ksel] = res3['lmfit_forbidden'].bic
        else:
            if 'all' in ztab['FAMILY']:
                ksel = ztab['FAMILY'] == 'all'
                ztab['BIC_OII'][ksel] = res3['lmfit_forbidden'].bic
    if 'CIII1907' in lines['LINE']:
        linelist = ['CIII1907','CIII1909']
        z = lines[lines['LINE']=='CIII1907']['Z'][0]
        res3 = pl.fit_lines(res['line_spec'], z, emcee=False, lines=linelist)
        if 'forbidden' in ztab['FAMILY']:
            ksel = ztab['FAMILY'] == 'forbidden'
            ztab['BIC_CIII'][ksel] = res3['lmfit_forbidden'].bic
        else:
            if 'all' in ztab['FAMILY']:
                ksel = ztab['FAMILY'] == 'all'
                ztab['BIC_CIII'][ksel] = res3['lmfit_forbidden'].bic        
    if 'LYALPHA' in lines['LINE']:
        linelist = ['LYALPHA']
        z = lines[lines['LINE']=='CIII1907']['Z'][0]
        res3 = pl.fit_lines(res['line_spec'], z, emcee=False, lines=linelist)        
        if 'lyalpha' in ztab['FAMILY']:
            ksel = ztab['FAMILY'] == 'lyalpha'
            ztab['BIC_LYALPHA'][ksel] = res3['lmfit_lyalpha'].bic   

    
    
