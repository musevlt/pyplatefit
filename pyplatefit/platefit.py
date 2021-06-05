import logging
import os
from astropy.table import Table, vstack, Column, MaskedColumn
from matplotlib import transforms
import numpy as np
from mpdaf.obj import Spectrum

from .cont_fitting import Contfit
from .eqw import EquivalentWidth
from .line_fitting import Linefit, plotline


__all__ = ('Platefit', 'fit_spec', 'plot_fit')

def muse_lsf(wave):
    # from UDF paper
    #fwhm = 5.835e-8 * wave**2 - 9.080e-4 * wave + 5.983
    # from DRS
    fwhm = 5.19939 - 0.000756746*wave + 4.93397e-08*wave**2
    return fwhm


class Platefit:
    """
    This class is a poorman version of Platefit.
    """

    def __init__(self, contpars={}, linepars={}, eqwpars={}, 
                 mcmcpars = dict(steps=0, nwalkers=0, save_proba=False),
                 minpars = dict(method='least_square', xtol=1.e-3)):
        """Initialise a Platefit object
        
        Parameters
        ----------     
        contpars: dictionary
          input parameters to be passed to `Contfit` constructor
          
        linepars: dictionary
          input parameters to be passed to `Linefit` constructor
          
        minpars: dictionary
          minimization parameters to be passed to `Linefit` constructor
          
        mcmcpars: dictionary
          EMCEE minimization parameters to be passed to `Linefit` constructor
               
        eqwpars: dictionary
          input parameters to be passed to `EquivalentWidth` constructor
          
           
        """
        self.logger = logging.getLogger(__name__)
        self.cont = Contfit(**contpars)
        self.line = Linefit(**linepars, minpars=minpars, mcmcpars=mcmcpars)
        self.eqw = EquivalentWidth(**eqwpars)

    def fit(self, spec, z, ziter=False, fitcont=True, fitlines=True, fitabs=False, eqw=True,
            lsf=muse_lsf, **kwargs):
        """Perform continuum and emission and absorption lines fit on a spectrum

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
           Fit and remove continuum prior to (absorption) line fitting.
        fitlines : bool
           Fit emission lines
        fitabs : bool
           Fit absorption lines
        eqw : bool
           Compute Equivalent Width
        lsf : function
            LSF model
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
            - result['poly_cont']: MPDAF spectrum, polynomial continuum estimation
            - result['abs_fit']: MPDAF spectrum, absorption line fit
            - result['spec_fit']: MPDAF spectrum, fitted line+continuum (+absorption) in
              observed frame
            - result['dcont']: return dictionary from fit_cont (see `fit_cont`)
            - result['dline']: returned dictionary from fit_lines (see `fit_lines`)

        """
        
        resfit = {'spec':spec}

        vdisp = kwargs.pop('vdisp', 80)

        if ziter and fitcont:
            self.logger.debug('Performing a first quick fit to refine the input redshift')
            resfit['iter_zinit'] = z
            rescont = self.fit_cont(spec, z, vdisp)
            linespec = rescont['line_spec'] 
            # set parameters to speed the fit
            kwargs1 = kwargs.copy()
            kwargs1['fit_all'] = True
            resline = self.fit_lines(linespec, z, lsf=lsf, **kwargs1)
            ztable = resline['ztable']
            if 'all' in ztable['FAMILY']:
                row = ztable[ztable['FAMILY']=='all'][0]
            elif 'lyalpha' in ztable['FAMILY']:
                row = ztable[ztable['FAMILY']=='lyalpha'][0]
            else:
                self.logger.error('No line fitting solution found')
                return None
            vel = row['VEL']
            z = row['Z']
            resfit['iter_z'] = z
            resfit['iter_vel'] = vel      
            self.logger.debug('Computed velocity offset %.1f km/s', vel)
            
        
        if fitcont:
            self.logger.debug('Fit continuum')
            rescont = self.fit_cont(spec, z, vdisp)
            linespec = rescont['line_spec']
            for key in ['cont_spec','cont_fit','line_spec']:
                if key in rescont.keys():
                    resfit[key] = rescont.pop(key)
            del rescont['spec']
            resfit['dcont'] = rescont
        else:
            resfit['line_spec'] = spec
            linespec = spec

        if fitlines:
            self.logger.debug('Fit emission lines')
            resline = self.fit_lines(linespec, z, lsf=lsf, **kwargs)

        if fitabs:
            self.logger.debug('Fit absorption lines')
            if fitlines:
                # remove emission lines
                linefit = resline['line_fit']
                spnoline = spec - linefit
            else:
                spnoline = spec

            resabs = self.fit_abslines(spnoline, z, lsf=lsf, **kwargs)

        if eqw and fitcont and fitlines:
            self.eqw.comp_eqw(spec, linespec, z, resline['lines'])
        if eqw and fitabs:
            self.eqw.comp_eqw(spec, resabs['abs_line'], z, resabs['lines'])

        if fitlines and fitabs:
            # add stacked lines and absorption lines result to resline dict
            resfit['ztable'] = vstack([resline.pop('ztable'),
                                       resabs.pop('ztable')])
            resfit['lines'] = vstack([resline.pop('lines'),
                                      resabs.pop('lines')])
        elif fitlines:
            resfit['ztable'] = resline.pop('ztable')
            resfit['lines'] = resline.pop('lines')
        elif fitabs:
            resfit['ztable'] = resabs.pop('ztable')
            resfit['lines'] = resabs.pop('lines')

        if fitlines or fitabs:
            resfit['lines'].sort('LBDA_REST')
            resfit['lines'].add_index('LINE')
            resfit['ztable'].add_index('FAMILY')

        if fitlines:
            resfit['line_spec'] = resline.pop('line_spec')
            resfit['line_fit'] = resline.pop('line_fit')
            resfit['line_initfit'] = resline.pop('line_initfit')
            resfit['dline'] = resline
        if fitcont:
            resfit['spec_fit'] = resfit['cont_spec']
        if fitlines:
            if fitcont:
                resfit['spec_fit'] += resfit['line_fit']
            else:
                resfit['spec_fit'] = resfit['line_fit']
        if fitabs:
            resfit['spec_fitp'] = resabs['abs_fit']
            if fitlines:
                resfit['spec_fitp'] += resfit['line_fit']
        if fitabs:
            resfit['abs_cont'] = resabs['abs_cont']
            resfit['abs_line'] = resabs['abs_line']
            resfit['abs_init'] = resabs['abs_init']
            resfit['abs_fit'] = resabs['abs_fit']
            if fitlines:
                resfit['dline']['lmfit_abs'] = resabs['lmfit_abs']
                resfit['dline']['abs_table_spec'] = resabs['table_spec']
            else:
                resfit['dline'] = resabs             

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

    
    def fit_lines(self, line, z, lsf=muse_lsf, **kwargs):
        """  
        Perform emission lines fit on a continuum subtracted spectrum 
    
        Parameters
        ----------
        line : mpdaf.obj.Spectrum
            continuum subtracted spectrum
        z : float
            reshift 
        lsf : function
            LSF model
        kwargs : keyword arguments
           Additional arguments passed to the `fit_lines` function.

       
        Returns
        -------
        res : dictionary
           See `Linefit.fit`
           
        """
        return self.line.fit(line, z, lsf=lsf, **kwargs)        
    
    def fit_abslines(self, spec, z, lsf=muse_lsf, **kwargs):
        """  
        Perform absorption lines fit on a continuum subtracted spectrum 
    
        Parameters
        ----------
        spec : mpdaf.obj.Spectrum
            input spectrum 
        z : float
            reshift 
        lsf : function
            LSF model
        kwargs : keyword arguments
           Additional arguments passed to the `fit_lines` function.

       
        Returns
        -------
        res : dictionary
           See `Linefit.fit`
           
        """
        return self.line.absfit(spec, z, lsf=lsf, **kwargs)         
        

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




def fit_spec(spec, z, fit_all=False, ziter=False, fitcont=True, fitlines=True, lines=None,
             major_lines=False, fitabs=False, vdisp=80, use_line_ratios=False, find_lya_vel_offset=False, dble_lyafit=False,
             mcmc_lya=False, mcmc_all=False,
             lsf=muse_lsf, eqw=True, trimm_spec=True, contpars={}, linepars={},
             mcmcpars=dict(steps=0, nwalkers=0, save_proba=False, progress=True),
             minpars=dict(method='least_square', xtol=1.e-3)):
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
    ziter : bool
      if True, a first emission line fit is performed to refine the redshift before a new continuum subtraction
      and a complete line fit is performed (to save computation time, eemce option is disactivated for the first fit),
      default false
    fitcont : bool
      If True, fit and subtract the continuum before performing an emission or absorption line fit (default: True)
    fitlines : bool
      If True, perform an emission line fit
    lines: list or astropy table
       the list specify the  MPDAF lines to use in the fit, while the astropy table
       is a replacement of the MPDAF line list table (see `Linefit.fit_lines` for more info)
    major_lines : bool
       if true, use only major lines as defined in MPDAF line list (default False).
    fitabs : bool
       if True, fit also absorption lines after the emission line fit
    vdisp : float
       fixed velocity dispersion in km/s (default 80 km/s) used in the continuum fit.
    use_line_ratios : bool
       if True, use constrain line ratios in fit (default False)
    find_lya_vel_offset : bool
       if True, perform an initial search for the lya velocity offset [deactivated for dble lya fit]
    dble_lyafit : bool
        if True, use a double asymetric gaussian model for the lya line fit
    mcmc_lya : bool
        if True, perform an MCMC LYALPHA fit after the first fit (default False)
    mcmc_all : bool
        if True, perform an MCMC fit after the first fit for all lines (default False)
    lsf : function
       LSF model to take into account the instrumental LSF (default to muse_lsf).
    eqw : bool
       if True compute equivalent widths (default True).
    trimm_spec : bool
       if True, trimmed spec around selected emission lines (default True).    
    contpars : dictionary
      Input parameters to pass to `Contfit` (default {})
    linepars : dictionary
      Input parameters to pass to `Linefit` (default {})
      
        - vel : (min,init,max), emi lines, bounds and init value for velocity offset (km/s), default (-500,0,500)
        - vdisp : (min,init,max), emi lines, bounds and init value for velocity dispersion (km/s), default (5,50,300)
        - velabs : (min,init,max), abs lines, bounds and init value for velocity offset (km/s), default (-500,0,500)
        - vdispabs : (min,init,max), abs lines, bounds and init value for velocity dispersion (km/s), default (5,50,300)
        - vdisp_lya : (min,init,max), bounds and init value for lya velocity dispersion (km/s), default (50,150,700)
        - gamma_lya : (min,init,max), bounds and init value for lya skewness parameter, default (-1,0,10)
        - gamma_2lya1 : (min,init,max), bounds and init value for lya left line skewness parameter, default (-10,-2,0)
        - gamma_2lya2 : (min,init,max), bounds and init value for lya right line skewness parameter, default (0,2,10)
        - sep_2lya : (min,init,max), bounds and init value for the 2 peak lya line separation (rest frame, km/s), default (80,500,1000)
        - windmax : float, maximum half size window in A to find peak values around initial wavelength value (default 10)
        - nstd_relsize : float, relative size (wrt to FWHM) of the wavelength window used for CHI2 line estimation (used in bootstrap only), default: 3.0
        - minsnr : float, minimum SNR to display line ID in plots (default 3.0)
        - line_ratios : list of tuples, list of line_ratios (see text), defaulted to [("CIII1907", "CIII1909", 0.6, 1.2), ("OII3726", "OII3729", 1.0, 2.0)]
   

    minpars : dictionary
      Input parameters to pass to minimize (lmfit) exemple:
      dict(method='nelder', options=dict(xatol=1.e-3)) 
      dict(method='least_square', xtol=1.e-3) [default]
      see https://docs.scipy.org/doc/scipy/reference/optimize.html for detailed info
      optional parameters for the given method,
      
    mcmcpars : dictionary
      Input parameters to pass to `Linefit` 
        - steps : (default 0 (10000 except for dble_lya 15000))
        - nwalkers : (default 0 = 25*npars)
        - save_proba: if True add P95 and P99 limits to fitted parameters (default False)
        
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
        
    The table of the lines found in the spectrum are given in the table lines. 
    The columns are:
    
      - FAMILY: the line family name (eg balmer)
      - LINE: The name of the line
      - LBDA_REST: The rest-frame position of the line in vacuum   
      - DNAME: The display name for the line (set to None for close doublets)
      - VEL: The velocity offset in km/s with respect to the initial redshift (rest frame)
      - VEL_ERR: The error in velocity offset in km/s 
      - Z: The fitted redshift in vacuum of the line (note for lyman-alpha the line peak is used)
      - Z_ERR: The error in fitted redshift of the line.
      - Z_INIT: The initial redshift 
      - VDISP: The fitted velocity dispersion in km/s (rest frame)
      - VDISP_ERR: The error in fitted velocity dispersion
      - VDINST: The instrumental velocity dispersion in km/s
      - FLUX: Flux in the line. The unit depends on the units of the spectrum.
      - FLUX_ERR: The fitting uncertainty on the flux value.
      - SNR: the SNR of the line
      - SKEW: The skewness parameter of the asymetric line (for Lyman-alpha line only).
      - SKEW_ERR: The uncertainty on the skewness (for Lyman-alpha line only).
      - SEP: The fitted lya rest frame peak separation (in km/s) (for the double lyman-alpha fit only)
      - SEP_ERR: The error in fitted lya rest frame peak separation (in km/s) (for the double lyman-alpha fit only)
      - LBDA_OBS: The fitted position the line peak in the observed frame
      - PEAK_OBS: The fitted peak of the line in the observed frame
      - LBDA_LEFT: The wavelength at the left of the peak with 0.5*peak value
      - LBDA_RIGHT: The wavelength at the rigth of the peak with 0.5*peak value     
      - FWHM_OBS: The full width at half maximum of the line in the observed frame 
      - NSTD: The log10 of the normalized standard deviation of the line fit 
      - LBDA_LNSTD: The wavelength at the left of the range used for NSTD estimation
      - LBDA_RNSTD: The wavelength at the right of the range used for NSTD estimation
      - EQW: The restframe line equivalent width 
      - EQW_ERR: The error in EQW
      - CONT_OBS: The continuum mean value in Observed frame
      - CONT: the continuum mean value in rest frame
      - CONT_ERR: the error in rest frame continuum
    
    The redshift table is saved in the table ztable
    The columns are:
    
      - FAMILY: the line family name
      - VEL: the velocity offset with respect to the original z in km/s
      - VEL_ERR: the error in velocity offset
      - Z: the fitted redshift (in vacuum)
      - Z_ERR: the error in redshift
      - Z_INIT: The initial redshift 
      - VDISP: The fitted velocity dispersion in km/s (rest frame)
      - VDISP_ERR: The error in fitted velocity dispersion
      - LINE: the emission line name with maximum SNR
      - SNRMAX: the maximum SNR
      - SNRSUM: the sum of SNR (all lines)
      - SNRSUM_CLIPPED: the sum of SNR (only lines above a MIN SNR (default 3))
      - NL: number of fitted lines
      - NL_CLIPPED: number of lines with SNR>SNR_MIN
      - NFEV: the number of function evaluation
      - RCHI2: the reduced Chi2 of the family lines fit 
      - METHOD: minimization method
      - STATUS: return status from minimization

    """
    logger = logging.getLogger(__name__)
    if isinstance(spec, str):
        spec = Spectrum(spec)    
    pl = Platefit(contpars=contpars, linepars=linepars, minpars=minpars, mcmcpars=mcmcpars)
    res = pl.fit(spec, z, fit_all=fit_all, ziter=ziter, fitcont=fitcont, fitlines=fitlines,
                 lines=lines, use_line_ratios=use_line_ratios, find_lya_vel_offset=find_lya_vel_offset,
                 dble_lyafit=dble_lyafit, lsf=lsf, eqw=eqw, vdisp=vdisp, trimm_spec=trimm_spec,
                 mcmc_lya=mcmc_lya, mcmc_all=mcmc_all,
                 fitabs=fitabs, major_lines=major_lines)
    return res   
                                       
def plot_fit(ax, result, line_only=False, abs_line=False, pcont=False,
             line=None, start=False, filterspec=0, 
             margin=50, legend=True, iden=True, label=True, minsnr=0, 
             infoline=None, infoz=None,
             labelpars={'dl':2.0, 'y':0.95, 'size':10, 'xl':0.05, 'yl':0.95, 'dyl':0.05, 'sizel':10}):
    """ 
    plot fitting results obtained with `fit_spec`
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes instance in which to draw the plot       
    result : dictionary 
        result of `fit_spec`
    line_only : bool
        plot the continuum subtracted spectrum and its fit
    abs_line : bool
        plot the absorption lines and continuum fit
    pcont : bool
        plot the polynomial and absorption line fit
    line : str
        name of the line to zoom on (if None display all the spectrum)
    start : bool
        plot also the initial guess before the fit (only if line is not None)
    filterspec : int
        width in pixel of the box-filter to apply on the input spectrum
    margin : float
        size in A to add on each side of the line for the zoom option
    legend : bool
        display the legend 
    iden : bool
        display the emission lines
    label : bool
        display the line name
    minsnr : float
        minimum SNR to display the line names
    infoline : list or None
        list of line parameters (from lines table) to display, only used with line=NAME
    infoz : list or None
        list of global parameters (from ztable) to display
    labelpars : dictionary
        parameters used for label display
        
          - dl: offset in wavelength
          - y: location in y (0-1)
          - size: font size
          - xl: x location for info
          - yl, dyl: y location and offset for info
          - sizel: info font size
    """
    logger = logging.getLogger(__name__)
    lines = result['lines']
    if line is not None and line not in lines['LINE']:
        logger.error('Line %s not found in table', line)
        return
    if line_only:
        # get and truncate spectra
        spline = result['line_spec']
        if filterspec > 0:
            spline = spline.filter(width=filterspec)           
        splinefit = result['line_fit']
        if start:
            spinitfit = result['line_initfit']
        if line is not None:
            spline = truncate_spec(spline, line, lines, margin)
            splinefit = truncate_spec(splinefit, line, lines, margin)
            if start:
                spinitfit = truncate_spec(spinitfit, line, lines, margin)
        # plot spectra
        rawlabel = f'cont subtracted (filtered {filterspec})' if filterspec > 0 else 'cont subtracted'
        spline.plot(ax=ax, label=rawlabel, color='k')
        splinefit.plot(ax=ax, color='r', drawstyle='default', label='fit')
        if start:
            spinitfit.plot(ax=ax, color='b', drawstyle='default', label='init fit')
    elif abs_line:
        # get and truncate spectra
        spraw = result['abs_init']
        if filterspec > 0:
            spraw = spraw.filter(width=filterspec)          
        spcont = result['abs_cont']
        spfit = result['abs_fit']
        if line is not None:
            spraw = truncate_spec(spraw, line, lines, margin)
            spcont = truncate_spec(spcont, line, lines, margin)
            spfit = truncate_spec(spfit, line, lines, margin)
        # plot spectra
        rawlabel = f'data (filtered {filterspec})' if filterspec > 0 else 'data'
        spraw.plot(ax=ax, color='k', label=rawlabel)
        spfit.plot(ax=ax, color='r', drawstyle='default', label='fit')
        spcont.plot(ax=ax, color='b', drawstyle='default', label='cont fit')                 
    else:
        # get and truncate spectra
        spraw = result['spec']
        if filterspec > 0:
            spraw = spraw.filter(width=filterspec)  
        try:
            spcont = result['cont_spec']
        except:
            spcont = 0.0*spraw 
        if pcont:
            spfit = result['spec_fitp']
            spcont = result['abs_cont']
            labcont = 'poly cont'
        else:
            spfit = result['spec_fit']
            if 'cont_spec' in result:
                spcont = result['cont_spec']
            else:
                spcont = None
            labcont = 'model cont'
        if line is not None:
            spraw = truncate_spec(spraw, line, lines, margin)
            if spcont is not None:
                spcont = truncate_spec(spcont, line, lines, margin)
            spfit = truncate_spec(spfit, line, lines, margin)
        # plot spectra
        rawlabel = f'data (filtered {filterspec})' if filterspec > 0 else 'data'
        spraw.plot(ax=ax, color='k', label=rawlabel)
        spfit.plot(ax=ax, color='r', drawstyle='default', label='fit')
        if spcont is not None:
            spcont.plot(ax=ax, color='b', drawstyle='default', label=labcont) 
        
    # display legend
    if legend:
        ax.legend()
        
    # remove x and y label
    ax.set_xlabel('')
    ax.set_ylabel('')
        
    # display lines
    if iden:
        if line_only:
            lmin,lmax = spline.get_range()
        else:
            lmin,lmax = spraw.get_range()
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)            
        for cline in lines[~lines['ISBLEND']]: 
            if (cline['LBDA_OBS']<lmin) or (cline['LBDA_OBS']>lmax):
                continue
            if (cline['DNAME'] == 'None') or (cline['SNR']<minsnr):
                color = 'red' if cline['FLUX'] > 0 else 'blue'
                ax.axvline(cline['LBDA_OBS'], color=color, alpha=0.2)
            else:
                color = 'red' if cline['FLUX'] > 0 else 'blue'
                y = labelpars['y'] if cline['FLUX'] > 0 else 1-labelpars['y']
                ax.axvline(cline['LBDA_OBS'], color=color, alpha=0.4)
                ax.text(cline['LBDA_OBS']+labelpars['dl'], y, cline['DNAME'], 
                        dict(fontsize=labelpars['size']), transform=trans)
                
        if (infoline is not None) or (infoz is not None):
            x,y,dy = labelpars['xl'],labelpars['yl'],labelpars['dyl']
                
        if (line is not None) and (infoline is not None):
            row = lines[lines['LINE']==line]
            if len(row) > 0:
                row = row[0]
                for key in infoline:
                    if key not in row.keys():
                        continue
                    val = row[key]
                    if isinstance(val, (float,np.float)):
                        tval = f"{key}:{val:.2f}"
                    elif val is None:
                        tval = f"{key}:None"
                    else:
                        tval = f"{key}:{val}"
                    ax.text(x, y, tval, fontsize=labelpars['sizel'], ha='left', transform=ax.transAxes, color='b')
                    y -= dy
                    
        if (line is not None) and (infoz is not None):
            row = lines[lines['LINE']==line]
            fam = row['FAMILY']
            ztab = result['ztable']
            row = ztab[ztab['FAMILY']==fam]
            if len(row) > 0:
                row = row[0]
                for key in infoz:
                    if key not in row.keys():
                        continue
                    val = row[key]
                    if isinstance(val, (float,np.float)):
                        tval = f"{key}:{val:.2f}"
                    elif val is None:
                        tval = f"{key}:None"
                    else:
                        tval = f"{key}:{val}"
                    ax.text(x, y, tval, fontsize=labelpars['sizel'], ha='left', transform=ax.transAxes, color='b')
                    y -= dy            
            

def print_res(res, lines):
    res0 = res['dline']
    names = [['Name','name'],['Min_bound','min'],['Max_bound','max'],['Init_value','init_value'],
             ['Value','value'],['Median','median_value'],['Init_Std','init_stderr'],['Std','stderr'],
             ['Acor','acor'],['Acor_ratio','acor_ratio'],
             ['Min_p99','min_p99'],['Min_p95','min_p95'],
             ['Max_p95','max_p95'],['Max_p99','max_p99'],]
    tab = Table(names=[e[0] for e in names], dtype=['S40']+13*['f4'], masked=True)
    for c in tab.columns:
        if c == 'Name':
            continue
        tab[c].format = '.3f'
    for line in lines:
        found = False
        for fam,par in res0.items():
            if fam[0:5] != 'lmfit':
                continue
            for key,val in par.params.items():
                if line not in key:
                    continue
                found = True
                break
            if found:
                break
        if found:
            for name,p in par.params.items():
                d = dict(Name=name)
                for col,key in names:
                    if col == 'Name':
                        continue
                    if hasattr(p,key):
                        d[col] = getattr(p, key)
                tab.add_row(d)
    return(tab)
                
                
def truncate_spec(spec, line, lines, margin):
    row = lines[lines['LINE']==line]
    if row is None:
        raise ValueError('line not found')
    row = row[0]
    l0 = row['LBDA_OBS']
    l1 = row['LBDA_LEFT'] - margin
    l2 = row['LBDA_RIGHT'] + margin
    sp = spec.subspec(lmin=l1, lmax=l2)
    return sp

 
            
        
    
    
    
    

    
    
