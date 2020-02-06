import logging
import os
from astropy.table import vstack, Column, MaskedColumn
from joblib import delayed, Parallel
from mpdaf.obj import Spectrum
from mpdaf.tools import progressbar
from matplotlib import transforms

from .cont_fitting import Contfit
from .eqw import EquivalentWidth
from .line_fitting import Linefit, plotline


__all__ = ('Platefit', 'fit_spec', 'plot_fit')


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
        resfit['lines'].add_index('LINE')
        resfit['ztable'] = resline.pop('ztable')
        resfit['ztable'].add_index('FAMILY')
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
             major_lines=False, vdisp=80, use_line_ratios=False, find_lya_vel_offset=True, dble_lyafit=False,
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
    lines: list or astropy table
       the list specify the  MPDAF lines to use in the fit, while the astropy table
       is a replacement of the MPDAF line list table (see `Linefit.fit_lines` for more info)
    major_lines : bool
       if true, use only major lines as defined in MPDAF line list (default False).
    vdisp : float
       velocity dispersion in km/s (default 80 km/s).
    use_line_ratios : bool
       if True, use constrain line ratios in fit (default False)
    find_lya_vel_offset : bool
       if True, perform an initial search for the lya velocity offset
    dble_lyafit : bool
        if True, use a double asymetric gaussian model for the lya line fit    
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
      
        - vel : (min,init,max), bounds and init value for velocity offset (km/s), default (-500,0,500)
        - vdisp : (min,init,max), bounds and init value for velocity dispersion (km/s), default (5,50,300)
        - vdisp_lya : (min,init,max), bounds and init value for lya velocity dispersion (km/s), default (50,150,700)
        - gamma_lya : (min,init,max), bounds and init value for lya skewness parameter, default (-1,0,10)
        - gamma_2lya1 : (min,init,max), bounds and init value for lya left line skewness parameter, default (-10,-2,0)
        - gamma_2lya2 : (min,init,max), bounds and init value for lya right line skewness parameter, default (0,2,10)
        - sep_2lya : (min,init,max), bounds and init value for the 2 peak lya line separation (rest frame, km/s), default (80,500,1000)
        - delta_vel : float, maximum excursion of Velocity Offset (km/s) with respect to the LSQ solution used for EMCEE fit, default 20
        - delta_vdisp : float, maximum excursion of Velocity Dispersion Offset (km/s) with respect to the LSQ solution used for EMCEE fit, default 10           
        - delta_gamma : float, maximum excursion of skewness lya parameter with respect to the LSQ solution used for EMCEE fit, default 0.1  
        - windmax : float, maximum half size window in A to find peak values around initial wavelength value (default 10)
        - xtol : float, relative error in the solution for the leastq fitting (default 1.e-4)
        - ftol : float, relative error in the sum of square for the leastsq fitting (default 1.e-6)
        - maxfev : int, max number of iterations for the leastsq fitting (default 1000)
        - steps : int, number of steps for the emcee minimisation (default 1000)
        - nwalkers : int, number of walkers for the emcee minimisation, if 0 it is computed as the nearest even number to 3*nvariables (default 0)
        - burn : int, number of first samples to remove from the analysis in emcee (default 100)
        - seed : None or int, random number seed (default None)
        - progress : bool, if True display progress bar during EMCEE computation (default False)
        - minsnr : float, minimum SNR to display line ID in plots (default 3.0)
        - line_ratios : list of tuples, list of line_ratios (see text), defaulted to [("CIII1907", "CIII1909", 0.6, 1.2), ("OII3726", "OII3729", 1.0, 2.0)] 
        
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
      - RCHI2: The reduced Chi2 of the fit
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
      - RCHI2: the reduced Chi2     

    """
    logger = logging.getLogger(__name__)
    if isinstance(spec, str):
        spec = Spectrum(spec)    
    pl = Platefit(contpars=contpars, linepars=linepars)
    logger.debug('Performing continuum and line fitting')
    res = pl.fit(spec, z, emcee=emcee, fit_all=fit_all, ziter=ziter, fitcont=fitcont,
                 lines=lines, use_line_ratios=use_line_ratios, find_lya_vel_offset=find_lya_vel_offset,    
                 dble_lyafit=dble_lyafit, lsf=lsf, eqw=eqw, vdisp=vdisp, trimm_spec=trimm_spec)
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
        z = lines[lines['LINE']=='LYALPHA']['Z'][0]
        res3 = pl.fit_lines(res['line_spec'], z, emcee=False, lines=linelist)        
        if 'lyalpha' in ztab['FAMILY']:
            ksel = ztab['FAMILY'] == 'lyalpha'
            ztab['BIC_LYALPHA'][ksel] = res3['lmfit_lyalpha'].bic   
            
            
def plot_fit(ax, result, line_only=False, line=None, start=False,
             filterspec=0,
             margin=50, legend=True, iden=True, label=True, minsnr=0, 
             labelpars={'dl':2.0, 'y':0.95, 'size':10}):
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
    line : str
        name of the line to zoom on (if None display all the spectrum)
    start : bool
        plot also the initial guess before the fit
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
    labelpars : dictionary
        parameters used for label display
        
          - dl: offset in wavelength
          - y: location in y (0-1)
          - size: font size
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
    else:
        # get and truncate spectra
        spraw = result['spec']
        if filterspec > 0:
            spraw = spraw.filter(width=filterspec)          
        spcont = result['cont_spec']
        spfit = result['spec_fit']
        if line is not None:
            spraw = truncate_spec(spraw, line, lines, margin)
            spcont = truncate_spec(spcont, line, lines, margin)
            spfit = truncate_spec(spfit, line, lines, margin)
        # plot spectra
        rawlabel = f'data (filtered {filterspec})' if filterspec > 0 else 'data'
        spraw.plot(ax=ax, color='k', label=rawlabel)
        spfit.plot(ax=ax, color='r', drawstyle='default', label='fit')
        spcont.plot(ax=ax, color='b', drawstyle='default', label='cont fit') 
        
    # display legend
    if legend:
        ax.legend()
        
    # display lines
    if iden:
        if line_only:
            lmin,lmax = spline.get_range()
        else:
            lmin,lmax = spraw.get_range()
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)            
        for cline in lines: 
            if (cline['LBDA_OBS']<lmin) or (cline['LBDA_OBS']>lmax):
                continue
            if (cline['DNAME'] == 'None') or (cline['SNR']<minsnr):
                ax.axvline(cline['LBDA_OBS'], color='red', alpha=0.2)
            else:
                ax.axvline(cline['LBDA_OBS'], color='red', alpha=0.4)
                ax.text(cline['LBDA_OBS']+labelpars['dl'], labelpars['y'], cline['DNAME'], 
                        dict(fontsize=labelpars['size']), transform=trans)

                
                
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
 
            
        
    
    
    
    

    
    
