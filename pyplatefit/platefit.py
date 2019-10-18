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

    def fit(self, spec, z, fitcont=True, eqw=True, **kwargs):
        """Perform continuum and emission lines fit on a spectrum

        Parameters
        ----------
        spec : mpdaf.obj.Spectrum
           continuum subtracted spectrum
        z : float
           initial reshift
        fitcont : bool
           Fit and remove continuum prior to line fitting.
        eqw : bool
           Compute Equivalent Width
        **kwargs : keyword arguments
           Additional arguments passed to `Linefit.fit` function.  

        Returns
        -------
        result : dict
        
            - result['linetable']: astropy line table (see `fit_lines`)
            - result['ztable']; astropy z table (see `fit_lines`)
            - result['spec']: MPDAF original spectrum
            - result['cont']: MPDAF spectrum, fitted continuum in observed
              frame
            - result['line']: MPDAF spectrum, continnum removed spectrum in
              observed frame
            - result['linefit']: MPDAF spectrum, fitted emission lines in
              observed frame
            - result['fit']: MPDAF spectrum, fitted line+continuum in
              observed frame
            - result['res_cont']: return dictionary from fit_cont (see `fit_cont`)
            - result['res_line']: returned ResultObject from fit_lines (see `fit_lines`)

        """
        if fitcont:
            vdisp = kwargs.pop('vdisp', 80)
            res_cont = self.fit_cont(spec, z, vdisp)
            linespec = res_cont['line_spec']
        else:
            res_cont = {}
            linespec = spec
        
        res_line = self.fit_lines(linespec, z, **kwargs)

        if eqw and fitcont:
            self.eqw.comp_eqw(spec, linespec, z,
                              res_line.linetable)

        return dict(
            linetable=res_line.linetable,
            ztable=res_line.ztable,
            cont=res_cont['cont_spec'] if fitcont else None,
            line=linespec,
            linefit=res_line.spec_fit,
            fit=res_line.spec_fit + res_cont['cont_spec'] if fitcont else res_line.spec_fit,
            spec=spec,
            res_cont=res_cont,
            res_line=res_line
        )

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

    def info(self, res, full_output=False):
        """ print fitting info form `fit` 
        
        Parameters
        ----------
        res:  
          results from `fit`
        full_output: bool
          if True write more information, default False      
        """
        self.logger.info('++++ Continuum fit info')
        self.cont.info(res['res_cont'])
        self.logger.info('++++ Line fit info')
        self.line.info(res['res_line'], full_output=full_output)

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

    def plot(self, ax, res, mode=2, start=False, iden=True, minsnr=0,
             line=None, margin=5, dplot={'dl': 2.0, 'y': 0.95, 'size': 10}):
        """Plot fit results.

        Parameters
        ----------
        ax : axes
           matplotlib ax
        res: dictionary
           result as given by `fit`
        mode: integer
           plot mode: 0=continuum, 1=line, 2=cont+line
           default: 2
        start: bool
           use for plot mode = 1, display start fitting value
           default False
        iden: bool
           if True, display line names (mode=1,2 only)
        minsnr: float
           minimum SNR to display the line name (mode=1,2 only)
           default: 0
        line: str
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
            self.line.plot(ax, res['res_line'], start=start, iden=iden,
                           minsnr=minsnr, line=line, margin=margin,
                           dplot=dplot)
        elif mode == 2:
            plotline(ax, res['spec'], res['fit'], res['cont'], None,
                     res['linetable'], iden=iden, minsnr=minsnr, line=line,
                     margin=margin, dplot=dplot)
        else:
            self.logger.error('unknown plot mode (0=cont,1=line,2=full)')


def fit_spec(spec, z, ziter=True, emcee=False, comp_bic=False, fitcont=True, lines=None, 
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
    ziter : bool
      If True, perform two successive emission line fits, first using all lines except Lya, 
      second with different line families (default True)
    emcee : bool
      if True perform a second fit using EMCEE to derive improved errors
      (note cpu intensive), default False.
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
    
        - result['linetable']: astropy line table (see `fit_lines`)
        - result['ztable']; astropy z table (see `fit_lines`)
        - result['spec']: MPDAF original spectrum
        - result['cont']: MPDAF spectrum, fitted continuum in observed
          frame
        - result['line']: MPDAF spectrum, continnum removed spectrum in
          observed frame
        - result['linefit']: MPDAF spectrum, fitted emission lines in
          observed frame
        - result['fit']: MPDAF spectrum, fitted line+continuum in
          observed frame
        - result['res_cont']: return dictionary from fit_cont (see `fit_cont`)
        - result['res_line']: returned ResultObject from fit_lines (see `fit_lines`)


    """
    logger = logging.getLogger(__name__)
    if isinstance(spec, str):
        spec = Spectrum(spec)    
    pl = Platefit(contpars=contpars, linepars=linepars)
    logger.debug('First iteration: Continuum and Line fit without line family selection except for lyman-alpha')
    res = pl.fit(spec, z, emcee=emcee if not ziter else False, fit_all=True, fitcont=fitcont,
                 lines=lines, use_line_ratios=use_line_ratios, find_lya_vel_offset=find_lya_vel_offset,    
                 lsf=lsf, eqw=eqw, vdisp=vdisp, trimm_spec=trimm_spec)
    ztab = res['ztable']
    if ziter:  
        logger.debug('Second iteration: Line fit for each line family')
        z = ztab[0]['Z']
        ltab = res['linetable']
        ztab1 = ztab[ztab['FAMILY'] == 'all'] 
        ltab1 = ltab[ltab['FAMILY'] == 'all'] 
        res = pl.fit(spec, z, emcee=emcee, fit_all=False)
        ztab = res['ztable']
    if comp_bic:
        for name in ['BIC_LYALPHA','BIC_OII','BIC_CIII']:
            ztab.add_column(MaskedColumn(name=name, dtype=float, length=len(ztab), mask=True))
            ztab[name].format = '.2f'
        # compute bic for individual lines (lya,oii,ciii)
        if 'OII3727' in res['linetable']['LINE']:
            lines = ['OII3727','OII3729']
            res3 = pl.fit_lines(res['line'], z, emcee=False, lines=lines, trimm_spec=True)
            if 'forbidden' in ztab['FAMILY']:
                ksel = ztab['FAMILY'] == 'forbidden'
                ztab['BIC_OII'][ksel] = res3.bic
            else:
                if 'all' in ztab['FAMILY']:
                    ksel = ztab['FAMILY'] == 'all'
                    ztab['BIC_OII'][ksel] = res3.bic
        if 'CIII1907' in res['linetable']['LINE']:
            lines = ['CIII1907','CIII1909']
            res3 = pl.fit_lines(res['line'], z, emcee=False, lines=lines, trimm_spec=True)
            if 'forbidden' in ztab['FAMILY']:
                ksel = ztab['FAMILY'] == 'forbidden'
                ztab['BIC_CIII'][ksel] = res3.bic
            else:
                if 'all' in ztab['FAMILY']:
                    ksel = ztab['FAMILY'] == 'all'
                    ztab['BIC_CIII'][ksel] = res3.bic            
        if 'LYALPHA' in res['linetable']['LINE']:
            lines = ['LYALPHA']
            res3 = pl.fit_lines(res['line'], z, emcee=False, lines=lines, trimm_spec=True)
            if 'lyalpha' in ztab['FAMILY']:
                ksel = ztab['FAMILY'] == 'lyalpha'
                ztab['BIC_LYALPHA'][ksel] = res3.bic
    
    if ziter:         
        ztab = vstack([ztab1, ztab])
        ltab = vstack([ltab1, res['linetable']])
        res['ztable'] = ztab
        res['linetable'] = ltab            
    else:
        res['ztable'] = ztab   
    
    return res
