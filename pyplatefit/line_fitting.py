"""
Copyright (c) 2018 CNRS / Centre de Recherche Astrophysique de Lyon

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import logging
from collections import OrderedDict
import os

from astropy import constants
from astropy import units as u
from astropy.table import Table
from astropy.table import MaskedColumn
from astropy.convolution import convolve, Box1DKernel
from lmfit.parameter import Parameters
from lmfit import Minimizer, report_fit
import numpy as np
from scipy.special import erf
from scipy.signal import argrelmin
from logging import getLogger
from matplotlib import transforms

from mpdaf.sdetect.linelist import get_emlines
from mpdaf.obj.spectrum import vactoair, airtovac

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", message="invalid value encountered in greater")

C = constants.c.to(u.km / u.s).value
SQRT2PI = np.sqrt(2*np.pi)

# Parameters used in the fitting
VEL_MIN, VEL_INIT, VEL_MAX = -500, 0, 500  # Velocity
VD_MIN, VD_INIT, VD_MAX = 10, 50, 300  # Velocity dispersion
VD_MAX_LYA = 700  # Maximum velocity dispersion for Lyman α
GAMMA_MIN, GAMMA_INIT, GAMMA_MAX = -1, 0, 10  # γ parameter for Lyman α
WINDOW_MAX = 30 # search radius in A for peak around starting wavelength
MARGIN_EMLINES = 0 # margin in pixel for emission line selection wrt to the spectrum edge

__all__ = ('Linefit', 'fit_lines')


class NoLineError(ValueError):
    """Error raised when there is no line to fit in the spectrum."""

    pass


class Linefit:
    """
    This class implement Emission Line fit
    """
    def __init__(self, vel=(-500,0,500), vdisp=(5,50,300), vdisp_lya_max=700, gamma_lya=(-1,0,10), 
                 windmax=10, xtol=1.e-4, ftol=1.e-6, maxfev=1000, minsnr=3.0,
                 steps=1000, nwalkers=0, min_nwalkers=100, burn=20, seed=None,
                 line_ratios = [
                    ("CIII1907", "CIII1909", 0.6, 1.2),
                    ("OII3727", "OII3729", 1.0, 2.0)
                    ]
                 ):
        """Initialize line fit parameters and return a Linefit object
          
        Parameters
        ----------
        vel : tuple of floats
          Minimum, init and maximum values of velocity offset in km/s (default: -500,0,500).
        vdisp: tuple of floats
          Minimum, init and maximum values of rest frame velocity dispersion in km/s (default: 5,80,300). 
        vdisp_lya_max : float
          Maximum velocity dispersion for the Lya line in km/s (default: 700).
        gamma_lya : tuple of floats
          Minimum, init and maximum values of the skeness parameter for the asymetric gaussain fit (default: -1,0,10).
        windmax : float 
          maximum half size window in A to find peak values around initial wavelength value (default: 10).
        xtol : float
          relative error in the solution for the leastq fitting (default: 1.e-4).
        ftol : float
          relative error in the sum of square for the leastsq fitting (default: 1.e-6).
        maxfev : int
          max number of iterations for the leastsq fitting (default: 1000).
        steps : int
          number of steps for the emcee minimisation (default: 1000).
        nwalkers : int
          number of walkers for the emcee minimisation,
          if 0, it is computed as the nearest even number to 3*nvariables (default: 0).
        burn : int
          number of first samples to remove from the analysis in emcee (default: 20).
        seed : None or int
          Random number seed (default: None)
        minsnr : float
          Minimum SNR to display line ID in plots (default: 3.0).
        line_ratios : list of tuples
          List of line_ratios (see text), defaulted to [("CIII1907", "CIII1909", 0.6, 1.2), ("OII3726", "OII3729", 1.0, 2.0)]      
              
        Return
        ------
        Linefit object
        
        """
    
             
        self.logger = getLogger(__name__)
        
        self.maxfev = maxfev # nb max of iterations (leastsq)
        self.xtol = xtol # relative error in the solution (leastq)
        self.ftol = ftol # relative error in the sum of square (leastsq)
        
        self.steps = steps # emcee steps
        self.nwalkers = nwalkers # emcee nwalkers (if 0 auto = even nearest to 3*nvarys)
        self.burn = burn # emcee burn
        self.seed = seed # emcee seed
        
        self.vel = vel # bounds in velocity km/s, rest frame
        self.vdisp = vdisp # bounds in velocity dispersion km/s, rest frame
        self.vdisp_lya_max = vdisp_lya_max # maximum lya velocity dispersion km/s, rest frame
        self.gamma = gamma_lya # bounds in lya asymmetry
        
        self.windmax = windmax # maximum half size window to find peak around initial wavelength value
        self.minsnr = minsnr # minium SNR for writing label of emission line in plot
        
        self.line_ratios = line_ratios # list of line ratios constraints
                         
        return
    

    def fit(self, line_spec, z, **kwargs):
        """
        perform line fit on a mpdaf spectrum
        
        Parameters
        ----------
        line_spec : `mpdaf.obj.Spectrum`
            input spectrum (must be continnum subtracted)
        z : float
            initial redshift
        **kwargs : keyword arguments
            Additional arguments passed to the `fit_lines` function.

        Returns
        -------
        res : dictionary
           See `fit_lines`,
           return in addition the following spectrum in the observed frame
           res.spec initial spectrum,
           res.init_fit spectrum of the starting solution for the line fit,
           res.spec_fit spectrum of the line fit

        """
 
        lsq_kws = dict(maxfev=self.maxfev, xtol=self.xtol, ftol=self.ftol)
        mcmc_kws = dict(steps=self.steps, nwalkers=self.nwalkers, burn=self.burn, seed=self.seed)
        fit_lws = dict(vel=self.vel, vdisp=self.vdisp, vdisp_lya_max=self.vdisp_lya_max, gamma=self.gamma, minsnr=self.minsnr)
        use_line_ratios = kwargs.pop('use_line_ratios', False)
        if use_line_ratios:
            line_ratios = self.line_ratios
        else:
            line_ratios = None        

        
        wave = line_spec.wave.coord(unit=u.angstrom).copy()
        data = line_spec.data   
        if line_spec.var is not None:
            std = np.sqrt(line_spec.var)
        else:
            std = None
    
        if std is not None:
            bad_points = std == 0
            std[bad_points] = np.inf
    
        try:
            unit_data = u.Unit(line_spec.data_header.get("BUNIT", None))
        except ValueError:
            unit_data = None
                   
        res = fit_lines(wave=wave, data=data, std=std, redshift=z,
                        unit_wave=u.angstrom, unit_data=unit_data, 
                        lsq_kws=lsq_kws, mcmc_kws=mcmc_kws, fit_lws=fit_lws,
                        **kwargs)
        
        tab = res.spectable    
        # convert wave to observed frame and air
        wave = tab['RESTWL']*(1 + z)
        wave = vactoair(wave)
        # add init and fitted spectra on the observed plane
        spfit = line_spec.clone()
        spfit.data = np.interp(line_spec.wave.coord(), wave, tab['LINEFIT'])
        spfit.data = spfit.data / (1 + z)
        spinit = line_spec.clone()
        spinit.data = np.interp(line_spec.wave.coord(), wave, tab['INIT'])
        spinit.data = spinit.data / (1 + z)
        
        res.spec = line_spec    
        res.spec_fit = spfit
        res.init_fit = spinit
                
        return res        
    
    def info(self, res, full_output=False):
        """ Print fit informations 
        
        Parameters
        ----------
        res : dictionary       
              results of `fit`
              
        full_output:
              boolean
              if True display more info,
              default False
        """
        if hasattr(res, 'ier'):
            self.logger.info(f"Line Fit (LSQ) Status: {res.ier} {res.message} Niter: {res.nfev}")
        else: 
            self.logger.info(f"Line Fit (EMCEE) Niter: {res.nfev}")
        self.logger.info(f"Line Fit RedChi2: {res.redchi:.2f} Bic: {res.bic:.2f}")
        self.logger.info(res.ztable)
 
        if full_output:
            report_fit(res)
            
    def plot(self, ax, res, start=False, iden=True, minsnr=0, line=None, margin=5,
             dplot={'dl':2.0, 'y':0.95, 'size':10}):
        """ plot fit results
        
        Parameters
        ----------
        ax: matplotlib.axes.Axes
           Axes instance in which to draw the plot
        
        res: dictionary 
             results of `fit`
             
        start: boolean
        
        """
        plotline(ax, res.spec, res.spec_fit, None, res.init_fit, res.linetable, start=start,
                 iden=iden, minsnr=minsnr, line=line, margin=margin, dplot=dplot)
  

def fit_lines(wave, data, std, redshift, *, unit_wave=None,
                       unit_data=None, vac=False, lines=None, line_ratios=None,
                       major_lines=False, emcee=False,
                       vel_uniq_offset=False, lsf=True, trimm_spec=True,
                       find_lya_vel_offset=True,
                       lsq_kws=None, mcmc_kws=None, fit_lws=None):
    """Fit lines from a set of arrays (wave, data, std) using lmfit.

    This function uses lmfit to perform a simple fit of know lines in
    a spectrum with a given redshift, to compute the flux and flux uncertainty
    of each line covered by the spectrum. It must be used on a continuum
    subtracted spectrum.

    All the emission lines known by `mpdaf.sdetect.linelist.get_emlines` are
    searched for, unless a list of line names or a table of lines is provided
    with the `lines` parameter. If `trim_spectrum` is set to true (default),
    the fit is done keeping only the parts of the spectrum around the expected
    lines.

    It is possible to add constraints in the ratio of some lines (for instance
    for doublets). These constraints are given in the line_ratios parameter and
    are expressed as a list of tuples (line_1 name, line_2 name, line_2 flux
    / line_1 flux minimum, flux ratio maximum). For instance, to constraint
    doublet fluxes and discriminate between single lines and doublets  in
    MUSE-UDF, we used:

        line_ratios = [
            ("CIII1907", "CIII1909", 0.6, 1.2),
            ("OII3726", "OII3729", 1.0, 2.0)
        ]

    That stands for:

        0.6 <= OII3729 / OII3726 <= 1.2
        1.0 <= CIII1909 / CIII1907 <= 2.0


    If the units of the wavelength and data axis are provided, the function
    will try to determine the unit of the computed fluxes.

    The table of the lines found in the spectrum are given as result.linetable. 
    The columns are:
    
      - LINE: The name of the line
      - LBDA_REST: The rest-frame position of the line in vacuum
      - FAMILY: the line family name (eg balmer)
      - DNAME: The display name for the line (set to None for close doublets)
      - VEL: The velocity offset in km/s with respect to the initial redshift (rest frame)
      - VEL_ERR: The error in velocity offset in km/s 
      - Z: The fitted redshift in vacuum of the line (note for lyman-alpha the line peak is used)
      - Z_ERR: The error in fitted redshift of the line.
      - Z_INIT: The initial redshift 
      - VDISP: The fitted velocity dispersion in km/s (rest frame)
      - VDISP_ERR: The error in fitted velocity dispersion
      - FLUX: Flux in the line. The unit depends on the units of the spectrum.
      - FLUX_ERR: The fitting uncertainty on the flux value.
      - SNR: the SNR of the line
      - SKEW: The skewness parameter of the asymetric line (for Lyman-alpha line only).
      - SKEW_ERR: The uncertainty on the skewness (for Lyman-alpha line only).
      - LBDA_OBS: The fitted position the line peak in the observed frame
      - LBDA_LEFT: The wavelength at the left of the peak with 0.5*peak value
      - LBDA_RIGHT: The wavelength at the rigth of the peak with 0.5*peak value
      - PEAK_OBS: The fitted peak of the line in the observed frame
      - FWHM_OBS: The full width at half maximum of the line in the observed frame 
      - VDINST: The instrumental velocity dispersion in km/s
      - EQW: The restframe line equivalent width 
      - EQW_ERR: The error in EQW
      - CONT_OBS: The continuum mean value in Observed frame
      - CONT: the continuum mean value in rest frame
      - CONT_ERR: the error in rest frame continuum
    
    The redshift table is saved in result.ztable
    The columns are:
    
      - FAMILY: the line family name
      - VEL: the velocity offset with respect to the original z in km/s
      - VEL_ERR: the error in velocity offset
      - Z: the fitted redshift (in vacuum)
      - Z_ERR: the error in redshift
      - Z_INIT: The initial redshift 
      - VDISP: The fitted velocity dispersion in km/s (rest frame)
      - VDISP_ERR: The error in fitted velocity dispersion
      - SNRMAX: the maximum SNR
      - SNRSUM: the sum of SNR (all lines)
      - SNRSUM_CLIPPED: the sum of SNR (only lines above a MIN SNR (default 3))
      - NL: number of fitted lines
      - NL_CLIPPED: number of lines with SNR>SNR_MIN
    
    
  
    Parameters
    ----------
    wave : array-like of floats
        The wavelength axis of the spectrum.
    data : array-like of floats
        The data (flux) axis of the spectrum.
    std : array-like of floats
        The standard deviation associated to the data. It can be set to None if
        it's not available.
    redshift : float
        Expected redshift of the spectrum.
    unit_wave : astropy.unit.Unit, optional
        The unit of the wavelength axis.
    unit_data : astropy.unit.Unit, optional
        The unit of the data axis.
    vac : boolean, optional
        If True, vacuum wavelength are expected; else air wavelength are
        expected. Default: False.
    lines : list of str or astropy.table.Table, optional
        To limit the searched for lines, it is possible to use this parameter
        to pass:
        - a list of line names as in MPDAF line list;
        - or an astropy table with the information on the lines to fit. The
          table must contain a LINE column with the name of the line and a LBDA
          column with the rest-frame, vaccuum wavelength of the lines.  Only the
          lines that are expected in the spectrum will be fitted. 
    line_ratios: list of (str, str, float, float) tuples or string
        List on line ratio constraints (line1 name, line2 name, line2/line1
        minimum, line2/line1 maximum.
    major_lines : boolean, optional
        If true, the fit is restricted to the major lines as defined in mpdaf line table (used only when lines is None, )
        default: False
    emcee : boolean, optional
        if true, errors and best fit is estimated with MCMC starting from the leastsq solution
        default: False
    vel_uniq_offset: boolean, optional
        if True, use same velocity offset for all lines (not recommended)
        if False, allow different velocity offsets between balmer, forbidden and resonnant lines
        default: false 
    lsf: boolean, optional
        if True, use LSF estimate to derive instrumental PSF, otherwise assume no LSF
        default: True
    trimm_spec: boolean, optional
        if True, mask unused wavelengths part
        default: True
    find_lya_vel_offset: boolean, optional
        if True, compute a starting velocity offset for lya on the data
    lsq_kws : dictionary with leasq parameters (see scipy.optimize.leastsq)
    mcmc_kws : dictionary with MCMC parameters (see emcee)
    fit_lws : dictionary with some default and bounds parameters

    Returns
    -------
    result : OrderedDict
        Dictionary containing several parameters from the fitting.
        result is the lmfit MinimizerResult object (see lmfit documentation)
        in addition it contains
        result.tabspec an astropy table with the following columns
        
            - RESTWL: restframe wavelength
            - FLUX: resframe data value
            - ERR: stddev of FLUX
            - INIT: init value for the fit
            - LINEFIT: final fit value
            
        result.linetable (see above)
        result.ztable (see above)
        

    Raises
    ------
    NoLineError: 
        when none of the fitted line can be on the spectrum at the
        given redshift.
        
    """
    logger = logging.getLogger(__name__)

    wave = np.array(wave)
    data = np.array(data)
    std = np.array(std) if std is not None else np.ones_like(data)
    
    # get defaut parameters for lines bounds
    if 'vel' in fit_lws.keys():
        VEL_MIN,VEL_INIT,VEL_MAX = fit_lws['vel']
    if 'vel' in fit_lws.keys():
        VD_MIN, VD_INIT, VD_MAX = fit_lws['vdisp']
    if 'vdisp_lya_max' in fit_lws.keys():
        VD_MAX_LYA = fit_lws['vdisp_lya_max']
    if 'gamma_lya' in fit_lws.keys():
        GAMMA_MIN, GAMMA_INIT, GAMMA_MAX = fit_lws['gamma_lya']
    if 'windmax' in fit_lws.keys():
        WINDOW_MAX = fit_lws['windmax']
    if 'minsnr' in fit_lws.keys():
            MIN_SNR = fit_lws['minsnr']    
    
    # convert wavelength in restframe and vacuum, scale flux and std
    wave_rest = airtovac(wave)/(1+redshift)
    data_rest = data*(1+redshift)
    std_rest = std*(1+redshift)
    
    # mask all points that have a std == 0
    mask = std <= 0
    if np.sum(mask) > 0:
        logger.debug('Masked %d points with std <= 0', np.sum(mask))
        wave_rest, data_rest, std_rest = wave_rest[~mask], data_rest[~mask], std_rest[~mask]    

    # Unit of the computed flux.
    if unit_wave is not None and unit_data is not None:
        # The flux is the integral of the data in the line profile.
        unit_flux = unit_data * unit_wave
    else:
        unit_flux = None


    # Fitting only some lines from mpdaf library.
    if type(lines) is list:
        lines_to_fit = lines
        lines = None
    else:
        lines_to_fit = None

    if lines is None:
        logger.debug("Getting lines from get_emlines...") 
        sel = 1 if major_lines else None
        lines = get_emlines(z=redshift, vac=True, sel=sel, margin=MARGIN_EMLINES,
                            lbrange=[wave.min(), wave.max()],
                            ltype="em", table=True, restframe=True)
        lines.rename_column("LBDA_OBS", "LBDA_REST")
        if lines_to_fit is not None:
            lines = lines[np.in1d(lines['LINE'], lines_to_fit)]
            if len(lines) < len(lines_to_fit):
                logger.debug(
                    "Some lines are not on the spectrum coverage: %s.",
                    ", ".join(set(lines_to_fit) - set(lines['LINE'])))
        lines['LBDA_EXP'] = (1 + redshift) * lines['LBDA_REST']
        if not vac:
            lines['LBDA_EXP'] = vactoair(lines['LBDA_EXP'])        
    else:
        lines['LBDA_EXP'] = (1 + redshift) * lines['LBDA']
        if not vac:
            lines['LBDA_EXP'] = vactoair(lines['LBDA_EXP'])
        lines = lines[
            (lines['LBDA_EXP'] >= wave.min()) &
            (lines['LBDA_EXP'] <= wave.max())
        ]

    # When there is no known line on the spectrum area.
    if not lines:
        raise NoLineError("There is no known line on the spectrum "
                          "coverage.")
    
    # Spectrum trimming
        # The window we keep around each line depend on the minimal and maximal
        # velocity (responsible for shifting the line), and on the maximal velocity
        # dispersion (responsible for the spreading of the line). We add a 3σ
        # margin.
    if trimm_spec:
        mask = np.full_like(wave, False, dtype=bool)  # Points to keep
        for row in lines:
            line_wave = row["LBDA_REST"]
            if row['LINE'] == "LYALPHA":
                vd_max = VD_MAX_LYA
            else:
                vd_max = VD_MAX
            wave_min = line_wave * (1 + VEL_MIN / C)
            wave_min -= 3 * wave_min * vd_max / C
            wave_max = line_wave * (1 + VEL_MAX / C)
            wave_max += 3 * wave_max * vd_max / C
            mask[(wave_rest >= wave_min) & (wave_rest <= wave_max)] = True
            #logger.debug("Keeping only waves in [%s, %s] for line %s.",
                         #wave_min, wave_max, row['LINE'])
        wave_rest, data_rest, std_rest = wave_rest[mask], data_rest[mask], std_rest[mask]
        logger.debug("%.1f %% of the spectrum is used for fitting.",
                     100 * np.sum(mask) / len(mask))
    

    # The fitting is done with lmfit. The model is a sum of Gaussian (or
    # skewed Gaussian), one per line.
    params = Parameters()  # All the parameters of the fit
    family_lines = {} # dictionary of family lines
    
    
    if vel_uniq_offset:
        families = [[[1,2,3],'all']]
    else:
        # fit different velocity and velocity dispersion for balmer and forbidden family 
        families = [[[1],'balmer'],[[2],'forbidden']]

    
    has_lya = False    
    for family_ids,family_name in families:
        ksel = lines['FAMILY']==family_ids[0]
        if len(family_ids)>1:
            for family_id in family_ids[1:]:
                ksel = ksel | (lines['FAMILY']==family_id)
        sel_lines = lines[ksel]
        if len(sel_lines) == 0:
            logger.debug('No %s lines to fit', family_name)
            continue        
        # remove LYALPHA if present
        if 'LYALPHA' in sel_lines['LINE']:
            has_lya = True
            sel_lines = sel_lines[sel_lines['LINE'] != 'LYALPHA']
        if len(sel_lines) == 0:
            logger.debug('No %s lines to fit', family_name)
            continue
        else:
            logger.debug('%d %s lines to fit', len(sel_lines), family_name)
        family_lines[family_name] = dict(lines=sel_lines['LINE'].tolist(), fun='gauss')
        params.add(f'dv_{family_name}', value=VEL_INIT, min=VEL_MIN, max=VEL_MAX)
        params.add(f'vdisp_{family_name}', value=VD_INIT, min=VD_MIN, max=VD_MAX)
        for line in sel_lines:         
            name = line['LINE']
            l0 = line['LBDA_REST']
            add_gaussian_par(params, family_name, name, l0, wave_rest, data_rest, redshift, lsf)
        if line_ratios is not None:
            # add line ratios bounds
            dlines = sel_lines[sel_lines['DOUBLET']>0]
            if len(dlines) == 0:
                continue
            add_line_ratio(params, line_ratios, dlines)
            

    if not vel_uniq_offset:   
        # fit a different velocity and velocity dispersion for each resonnant lines(or doublet)
        family_id = 3; family_name = 'resonnant'
        sel_lines = lines[lines['FAMILY']==family_id]
        if len(sel_lines) == 0:
            logger.debug('No %s lines to fit', family_name)
        else:
            if 'LYALPHA' in sel_lines['LINE']:
                has_lya = True
                sel_lines = sel_lines[sel_lines['LINE'] != 'LYALPHA']     
            if len(sel_lines) == 0:
                logger.debug('No %s lines to fit', family_name)
            else:   
                logger.debug('%d %s lines to fit', len(sel_lines), family_name)
                doublets = sel_lines[sel_lines['DOUBLET']>0]
                singlets = sel_lines[sel_lines['DOUBLET']==0]
                if len(singlets) > 0:
                    for line in singlets:
                        if line['LINE'] == 'LYALPHA':
                            has_lya = True
                            continue
                        # we fit a gaussian
                        fname = line['LINE'].lower()
                        family_lines[fname] = dict(lines=[line['LINE']], fun='gauss')
                        params.add(f'dv_{fname}', value=VEL_INIT, min=VEL_MIN, max=VEL_MAX)
                        params.add(f'vdisp_{fname}', value=VD_INIT, min=VD_MIN, max=VD_MAX)
                        name = line['LINE']
                        l0 = line['LBDA_REST']
                        add_gaussian_par(params, fname, name, l0, wave_rest, data_rest, redshift, lsf)
                if len(doublets) > 0:
                    ndoublets = np.unique(doublets['DOUBLET'])
                    for dlbda in ndoublets:
                        dlines = doublets[np.abs(doublets['DOUBLET']-dlbda) < 0.01]
                        fname = str(dlines['LINE'][0]).lower()
                        family_lines[fname] = dict(lines=dlines['LINE'], fun='gauss')
                        params.add(f'dv_{fname}', value=VEL_INIT, min=VEL_MIN, max=VEL_MAX)
                        params.add(f'vdisp_{fname}', value=VD_INIT, min=VD_MIN, max=VD_MAX)              
                        for line in dlines:
                            name = line['LINE']
                            l0 = line['LBDA_REST']
                            add_gaussian_par(params, fname, name, l0, wave_rest, data_rest, redshift, lsf)
                        if line_ratios is not None:
                            # add line ratios bounds
                            add_line_ratio(params, line_ratios, dlines)
    if has_lya:
        family_lines['lyalpha'] = {'lines':['LYALPHA'], 'fun':'asymgauss'}
        name = 'LYALPHA'
        line = lines[lines['LINE']==name][0]
        # we fit an asymmetric line
        fname = name.lower()
        family_lines[fname] = dict(lines=[line['LINE']], fun='asymgauss')
        l0 = line['LBDA_REST']
        if find_lya_vel_offset:
            vel_init = get_lya_vel_offset(l0, wave_rest, data_rest)
            logger.debug('Computed Lya init velocity offset: %.2f', vel_init)
        else:
            vel_init = VEL_INIT
        params.add(f'dv_{fname}', value=vel_init, min=vel_init+VEL_MIN, max=vel_init+VEL_MAX)
        params.add(f'vdisp_{fname}', value=VD_INIT, min=VD_MIN, max=VD_MAX_LYA) 
        add_asymgauss_par(params, fname, name, l0, wave_rest, data_rest, redshift, lsf)
        logger.debug('Lyman alpha asymetric line fit')
             
    # Perform LSQ fit    
    minner = Minimizer(residuals, params, fcn_args=(wave_rest, data_rest, std_rest, family_lines, redshift, lsf))
    
    logger.debug('Leastsq fitting with ftol: %.0e xtol: %.0e maxfev: %d',lsq_kws['ftol'],lsq_kws['xtol'],lsq_kws['maxfev'])
    result = minner.minimize(**lsq_kws)
    logger.debug('%s after %d iterations, redChi2 = %.3f',result.message,result.nfev,result.redchi)
    
    # Perform MCMC
    if emcee:  
        # check if nwalkers is in auto mode
        if ('nwalkers' in mcmc_kws) and (mcmc_kws['nwalkers']==0):
            # nearest even number to 3*nb of variables 
            mcmc_kws['nwalkers'] = int(np.ceil(3*result.nvarys/2)*2)
        logger.debug('Error estimation using EMCEE with nsteps: %d nwalkers: %d burn: %d',mcmc_kws['steps'],mcmc_kws['nwalkers'],mcmc_kws['burn'])
        result = minner.emcee(params=result.params, is_weighted=True, float_behavior='chi2', **mcmc_kws)
        logger.debug('End EMCEE after %d iterations, redChi2 = %.3f',result.nfev,result.redchi)
    
    # save input data, initial and best fit (in rest frame) in the table_spec table
    data_init = model(params, wave_rest, family_lines, redshift, lsf)
    data_fit = model(result.params, wave_rest, family_lines, redshift, lsf)
     
    tabspec = Table(data=[wave_rest,data_rest,std_rest,data_init,data_fit], 
                names=['RESTWL','FLUX','ERR','INIT','LINEFIT'])
    result.spectable = tabspec

   
    # fill the lines table with the fit results
    lines.remove_columns(['LBDA_LOW','LBDA_UP','TYPE','DOUBLET','LBDA_EXP','FAMILY'])
    colnames = ['VEL','VEL_ERR','Z','Z_ERR','Z_INIT','VDISP','VDISP_ERR',
                    'FLUX','FLUX_ERR','SNR','SKEW','SKEW_ERR','LBDA_OBS',
                    'PEAK_OBS','LBDA_LEFT','LBDA_RIGHT','FWHM_OBS', 'RCHI2'] 
    if lsf:
        colnames.append('VDINST')
    for colname in colnames:
        lines.add_column(MaskedColumn(name=colname, dtype=np.float, length=len(lines), mask=True))
    lines.add_column(MaskedColumn(name='FAMILY', dtype='U20', length=len(lines), mask=True), index=0)
    for colname in colnames:
        lines[colname].format = '.2f'
    lines['Z'].format = '.5f'
    lines['Z_INIT'].format = '.5f'
    lines['Z_ERR'].format = '.2e'

#   set ztable for global results by family 
    ftab = Table()
    ftab.add_column(MaskedColumn(name='FAMILY', dtype='U20', mask=True))
    colnames =  ['VEL','VEL_ERR','Z','Z_ERR','Z_INIT','VDISP','VDISP_ERR','SNRMAX','SNRSUM','SNRSUM_CLIPPED']
    for colname in colnames:
        ftab.add_column(MaskedColumn(name=colname, dtype=np.float, mask=True))
    for colname in colnames:
        ftab[colname].format = '.2f'
    for colname in ['NL','NL_CLIPPED']:
            ftab.add_column(MaskedColumn(name=colname, dtype=np.int, mask=True))  
    ftab['Z'].format = '.5f'
    ftab['Z_ERR'].format = '.2e'
    ftab['Z_INIT'].format = '.5f'
        
    par = result.params
    zf = 1+redshift
    for fname,fdict in family_lines.items():
        dv = par[f"dv_{fname}"].value 
        dv_err = par[f"dv_{fname}"].stderr if par[f"dv_{fname}"].stderr is not None else np.nan
        vdisp = par[f"vdisp_{fname}"].value 
        vdisp_err = par[f"vdisp_{fname}"].stderr if par[f"vdisp_{fname}"].stderr is not None else np.nan
        fwhm = 2.355*vdisp
        fwhm_err = 2.355*vdisp_err
        fun = fdict['fun']
        snr = []
        zlist = []
        dvlist = []
        for row in lines:
            name = row['LINE']
            if name not in fdict['lines']:
                continue
            # in rest frame
            flux = par[f"{name}_{fun}_flux"].value 
            if par[f"{name}_{fun}_flux"].expr is None:
                flux_err = par[f"{name}_{fun}_flux"].stderr if par[f"{name}_{fun}_flux"].stderr is not None else np.nan
            else: #try to estimate std using the constrain (but seems to give too high std values)
                expr = par[f"{name}_{fun}_flux"].expr
                fact = par[expr.split(' ')[-1]]
                flux2 = par[expr.split(' ')[0]]
                flux_err = fact.value*flux2.stderr if flux2.stderr is not None else np.nan
                # fact.value*flux2.stderr + fact.stderr*flux2.value 
            row['FAMILY'] = fname
            row['VEL'] = dv
            row['VEL_ERR'] = dv_err
            row['Z'] = redshift + dv/C
            row['Z_INIT'] = redshift
            row['Z_ERR'] = dv_err/C
            row['FLUX'] = flux
            row['FLUX_ERR'] = flux_err
            row['SNR'] = row['FLUX']/row['FLUX_ERR']
            row['VDISP'] = vdisp
            row['VDISP_ERR'] = vdisp_err
            # in observed frame
            row['LBDA_OBS'] = row['LBDA_REST']*(1+row['Z'])
            if not vac:
                row['LBDA_OBS'] = vactoair(row['LBDA_OBS'])
            sigma = get_sigma(vdisp, row['LBDA_OBS'], row['Z'], lsf, restframe=False) 
            if lsf:
                row['VDINST'] = complsf(row['LBDA_OBS'], kms=True)      
            peak = flux/(SQRT2PI*sigma)
            row['PEAK_OBS'] = peak
            row['FWHM_OBS'] = 2.355*sigma 
            row['LBDA_LEFT'] = row['LBDA_OBS'] - 0.5*row['FWHM_OBS']
            row['LBDA_RIGHT'] = row['LBDA_OBS'] + 0.5*row['FWHM_OBS']
            if fun == 'asymgauss':
                skew = par[f"{name}_{fun}_asym"].value 
                row['SKEW'] = skew 
                row['SKEW_ERR'] = par[f"{name}_{fun}_asym"].stderr if par[f"{name}_{fun}_asym"].stderr is not None else np.nan
                # compute peak location and peak value in rest frame 
                l0 = row['LBDA_REST']  
                swave_rest = np.linspace(l0-50,l0+50,1000)
                #ksel = np.abs(wave_rest-l0)<50
                #swave_rest = wave_rest[ksel]
                vmodel_rest = model_asymgauss(redshift, lsf, l0, flux, skew, vdisp, dv, swave_rest)
                kmax = np.argmax(vmodel_rest)    
                l1 = swave_rest[kmax]
                left_rest,right_rest = rest_fwhm_asymgauss(swave_rest, vmodel_rest)
                # these position is used for redshift and dv
                dv = C*(l1-l0)/l0
                row['VEL'] = dv
                row['Z'] = redshift + dv/C  
                # compute the peak value and convert it to observed frame    
                row['PEAK_OBS'] = np.max(vmodel_rest)/(1+row['Z'])
                # save peak position in observed frame
                row['LBDA_OBS'] = vactoair(l1*(1+row['Z']))
                row['LBDA_LEFT'] = vactoair(left_rest*(1+row['Z']))
                row['LBDA_RIGHT'] = vactoair(right_rest*(1+row['Z']))
                row['FWHM_OBS'] = row['LBDA_RIGHT'] - row['LBDA_LEFT'] 
            zlist.append(row['Z'])
            dvlist.append(row['VEL'])
            snr.append(row['SNR'])
                
        snr = np.array(snr)
        nline = len(snr)
        if np.all(np.isnan(snr)):
            sum_snr_clipped = np.nan
            sum_snr = np.nan
            snrmax = np.nan
            nline_snr_clipped = 0
        else:       
            snrmax = np.nanmax(snr)  
            sum_snr = np.sqrt(np.nansum(snr**2))
            snr = snr[snr>MIN_SNR]
            nline_snr_clipped = len(snr)
            if nline_snr_clipped == 0:
                sum_snr_clipped = np.nan
            else:
                sum_snr_clipped = np.sqrt(np.sum(snr**2))
        dv = np.mean(dvlist)
        zm = np.mean(zlist)
        ftab.add_row(dict(FAMILY=fname, VEL=dv, VEL_ERR=dv_err, VDISP=vdisp, VDISP_ERR=vdisp_err, 
                          Z=zm, Z_ERR=dv_err/C, SNRMAX=snrmax, Z_INIT=redshift,
                          NL=nline, NL_CLIPPED=nline_snr_clipped, SNRSUM=sum_snr, SNRSUM_CLIPPED=sum_snr_clipped))
        
        
    # save line table
    result.linetable = lines
    # save z table results
    result.ztable = ftab
   
    
    
    return result

def add_gaussian_par(params, family_name, name, l0, wave, data, z, lsf):
    params.add(f"{name}_gauss_l0", value=l0, vary=False)  
    ksel = np.abs(wave-l0) < WINDOW_MAX
    vmax = data[ksel].max()
    sigma = get_sigma(VD_INIT, l0, z, lsf, restframe=True)                  
    flux = SQRT2PI*sigma*vmax
    params.add(f"{name}_gauss_flux", value=flux, min=0)
    
def add_asymgauss_par(params, family_name, name, l0, wave, data, z, lsf):
    params.add(f"{name}_asymgauss_l0", value=l0, vary=False)  
    ksel = np.abs(wave-l0) < WINDOW_MAX
    vmax = data[ksel].max()
    sigma = get_sigma(VD_INIT, l0, z, lsf, restframe=True)  
    flux = SQRT2PI*sigma*vmax
    params.add(f"{name}_asymgauss_flux", value=flux, min=0)
    params.add(f"{name}_asymgauss_asym", value=GAMMA_INIT, min=GAMMA_MIN, max=GAMMA_MAX)
    
def get_lya_vel_offset(l0, wave, data, box_filter=3):
    ksel = np.abs(wave-l0) < WINDOW_MAX
    sel_data = data[ksel]
    if box_filter > 0:
        fsel_data =  convolve(sel_data, Box1DKernel(box_filter))
    else:
        fsel_data = sel_data
    kmax = fsel_data.argmax()
    lmax = wave[ksel][kmax]
    vel_off = (lmax/l0 - 1)*C
    return vel_off
    
    
    
def add_line_ratio(params, line_ratios, dlines):
    for line1,line2,ratio_min,ratio_max in line_ratios:
        if (line1 in dlines['LINE']) and (line2 in dlines['LINE']):
            params.add("%s_to_%s_factor" % (line1, line2), min=ratio_min,
                       max=ratio_max, value=0.5*(ratio_min+ratio_max))
            params['%s_gauss_flux' % line2].expr = (
                "%s_gauss_flux * %s_to_%s_factor" % (line1, line1, line2)
            ) 
            
def get_sigma(vdisp, l0, z, lsf=True, restframe=True):
    sigma = vdisp*l0/C
    if lsf:
        if restframe:
            l0obs = l0*(1+z)
            siginst = complsf(l0obs)/(1+z)
        else:
            siginst = complsf(l0)
        sigma = np.sqrt(sigma**2+siginst**2) 
    return sigma

def model(params, wave, lines, z, lsf=True):
    """ wave is rest frame wavelengths """
    model = 0
    for name,ldict in lines.items():
        vdisp = params[f"vdisp_{name}"].value
        dv = params[f"dv_{name}"].value
        for line in ldict['lines']:
            if ldict['fun']=='gauss':
                flux = params[f"{line}_gauss_flux"].value
                l0 = params[f"{line}_gauss_l0"].value
                model += model_gauss(z, lsf, l0, flux, vdisp, dv, wave)
            elif ldict['fun']=='asymgauss':
                flux = params[f"{line}_asymgauss_flux"].value
                l0 = params[f"{line}_asymgauss_l0"].value
                beta = params[f"{line}_asymgauss_asym"].value  
                model += model_asymgauss(z, lsf, l0, flux, beta, vdisp, dv, wave)         
            else:
                logger.error('Unknown function %s', fun)
                raise ValueError
    return model

def model_asymgauss(z, lsf, l0, flux, beta, vdisp, dv, wave):
    l1 = l0*(1+dv/C)
    sigma = get_sigma(vdisp, l1, z, lsf, restframe=True)               
    peak = flux/(SQRT2PI*sigma)
    model = asymgauss(peak, l1, sigma, beta, wave)
    return model

def model_gauss(z, lsf, l0, flux, vdisp, dv, wave):
    l1 = l0*(1+dv/C)
    sigma = get_sigma(vdisp, l1, z, lsf, restframe=True)               
    peak = flux/(SQRT2PI*sigma)
    model = gauss(peak, l1, sigma, wave)
    return model

def complsf(wave, kms=False):
    # compute estimation of LSF in A
    # from UDF paper
    #fwhm = 5.835e-8 * wave**2 - 9.080e-4 * wave + 5.983
    # from DRS
    fwhm = 5.19939 - 0.000756746*wave + 4.93397e-08*wave**2
    sigma = fwhm/2.355
    if kms:
        sigma = sigma*C/wave
    return sigma
    
    
def residuals(params, wave, data, std, lines, z, lsf=True):
    vmodel = model(params, wave, lines, z, lsf)
    res = (vmodel - data)/std 
    return res
    
def gauss(peak, l0, sigma, wave):
    g = peak*np.exp(-(wave-l0)**2/(2*sigma**2))
    return g

def asymgauss(peak, l0, sigma, gamma, wave):
    dl = wave - l0
    g = peak*np.exp(-dl**2/(2*sigma**2))
    f = 1 + erf(gamma*dl/(1.4142135623730951*sigma))
    h = f*g
    return h

def rest_fwhm_asymgauss(lbda, flux):
    g = flux/flux.max()
    kmax = g.argmax()
    l1 = None
    for k in range(kmax,0,-1):
        if g[k] < 0.5:
            l1 = np.interp(0.5, [g[k],g[k+1]],[lbda[k],lbda[k+1]])
            break
    if l1 is None:
        return None
    l2 = None
    for k in range(kmax,len(lbda),1):
        if g[k] < 0.5:
            l2 = np.interp(0.5, [g[k],g[k-1]],[lbda[k],lbda[k-1]])
            break
    if l2 is None:
        return None 
    return l1,l2
    
def mode_skewedgaussian(location, scale, shape):
    """Compute the mode of a skewed Gaussian.

    The centre parameter of the SkewedGaussianModel from lmfit-py is not the
    position of the peak, it's the location of the probability distribution
    function (PDF): i.e. the mean of the underlying Gaussian.

    There is no analytic expression for the mode (position of the maximum) but
    the Wikipedia page (https://en.wikipedia.org/wiki/Skew_normal_distribution)
    gives a "quite accurate" approximation.

    Parameters
    ----------
    location : float
        Location of the PDF. This is the`center` parameter from lmfit.
    scale : float
        Scale of the PDF. This is the `sigma` parameter from lmfit.
    shape : float
        Shape of the PDF. This is the `gamma` parameter from lmfit.

    Returns
    -------
    float: The mode of the PDF.

    """
    # If the shape is 0, this is a Gaussian and the mode is the location.
    if shape == 0:
        return location

    delta = scale / np.sqrt(1 + scale ** 2)
    gamma_1 = ((4 - np.pi) / 2) * (
        (delta * np.sqrt(2 / np.pi) ** 3) /
        (1 - 2 * delta ** 2 / np.pi) ** 1.5
    )

    mu_z = delta * np.sqrt(2 / np.pi)
    sigma_z = np.sqrt(1 - mu_z ** 2)
    m_0 = (mu_z - gamma_1 * sigma_z / 2 - np.sign(shape) / 2 *
           np.exp(-2 * np.pi / np.abs(shape)))

    return location + scale * m_0


def measure_fwhm(wave, data, mode):
    """Measure the FWHM on the curve.

    This function is used to measure the full width at half maximum (FWHM) of
    a line identified by its mode (wavelength of its peak) on the best fitting
    model. It is used for the Lyman α line for which we don't have a formula to
    compute it.

    Note: the data must be continuum subtracted.

    Parameters
    ----------
    wave : array of floats
        The wavelength axis of the spectrum.
    data : array of floats
        The data from lmfit best fit.
    mode : float
        The value of the mode, in the same unit as the wavelength.

    Returns
    -------
    float
        Full width at half maximum in the same value as the wavelength.

    """
    # In the case of lines with a strong asymmetry, it may be difficult to
    # measure the FWHM at the resolution of the spectrum. We multiply its
    # resolution by 10 to be able to measure the FWHM at sub-pixel level.
    wave2 = np.linspace(np.min(wave), np.max(wave), 10 * len(wave))
    data = np.interp(wave2, wave, data)
    wave = wave2

    mode_idx = np.argmin(np.abs(wave - mode))
    half_maximum = data[mode_idx] / 2

    # If the half maximum in 0, there is no line.
    if half_maximum == 0:
        return np.nan

    # If the half_maximum is negative, that means that the line is identified
    # in absorption. We can nevertheless try to measure a FWHM but on the
    # opposite of the data because of the way we measure it.
    if half_maximum < 0:
        half_maximum = -half_maximum
        data = -data

    # There may be several lines in the spectrum, so it may cross several times
    # the half maximum line and we can't use argmin on the absolute value of
    # the difference to the half maximum because it may be lower - but still
    # near zero - for another line. Instead, we are looking for the local
    # minimums and take the nearest to the mode position.
    # In case of really strong asymmetry, we may need to take the mode
    # position.
    try:
        hm1_idx = argrelmin(np.abs(data[:mode_idx + 1] - half_maximum))[0][-1]
    except IndexError:
        hm1_idx = mode_idx
    try:
        hm2_idx = (argrelmin(np.abs(data[mode_idx:] - half_maximum))[0][0] +
                   mode_idx)
    except IndexError:
        hm2_idx = mode_idx

    return wave[hm2_idx] - wave[hm1_idx]

def plotline(ax, spec, spec_fit, spec_cont, init_fit, table, start=False, iden=True, minsnr=0, line=None, margin=5,
         dplot={'dl':2.0, 'y':0.95, 'size':10}):
    sp = spec
    spfit = spec_fit
    spinit = init_fit
    spcont = spec_cont
    if line is not None:
        if line not in table['LINE']:
            self.logger.error('Line %s not found in table', line)
            return
        row = table[table['LINE']==line][0]
        l0 = row['LBDA_OBS']
        l1 = l0 - 3*row['FWHM_OBS'] - margin
        l2 = l0 + 3*row['FWHM_OBS'] + margin
        sp = spec.subspec(lmin=l1, lmax=l2)
        spfit = spec_fit.subspec(lmin=l1, lmax=l2)
        if start:
            spinit = init_fit.subspec(lmin=l1, lmax=l2)
        if spcont is not None:
            spcont = spec_cont.subspec(lmin=l1, lmax=l2)
        
    # plot the continuum removed spectrum
    sp.plot(ax=ax, label='data', color='k')
    # plot the line fit and eventually the fir initialization
    spfit.plot(ax=ax, color='r', drawstyle='default', label='fit') 
    if start:
        spinit.plot(ax=ax, color='g', drawstyle='default', label='init', alpha=0.4) 
    if spcont is not None:
        spcont.plot(ax=ax, color='g', drawstyle='default', label='cont', alpha=0.8)
    if iden:
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)            
        for line in table:               
            if (line['DNAME'] == 'None') or (line['SNR']<minsnr):
                ax.axvline(line['LBDA_OBS'], color='blue', alpha=0.2)
            else:
                ax.axvline(line['LBDA_OBS'], color='blue', alpha=0.4)
                ax.text(line['LBDA_OBS']+dplot['dl'], dplot['y'], line['DNAME'], dict(fontsize=dplot['size']), transform=trans)
        
    ax.legend()
    name = getattr(spec, 'filename', '')
    if name != '':
        name = os.path.basename(name)
    ax.set_title(f'Emission Lines Fit {name}')  
