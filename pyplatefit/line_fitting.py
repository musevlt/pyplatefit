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
import os

import copy
import more_itertools as mit
from astropy import constants
from astropy import units as u
from astropy.table import Table
from astropy.table import MaskedColumn
from astropy.convolution import convolve, Box1DKernel
from lmfit.parameter import Parameters
from lmfit import Minimizer
import numpy as np
from scipy.special import erf
from logging import getLogger
from matplotlib import transforms 
import emcee

from .linelist import get_lines
from mpdaf.obj.spectrum import vactoair, airtovac



import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", message="invalid value encountered in greater")
warnings.filterwarnings("ignore", message="Initial state is not linearly independent and it will not allow a full exploration of parameter space")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")

C = constants.c.to(u.km / u.s).value
SQRT2PI = np.sqrt(2*np.pi)

# Parameters used in the fitting
VEL_MIN, VEL_INIT, VEL_MAX = -500, 0, 500  # Velocity
VD_MIN, VD_INIT, VD_MAX = 10, 50, 300  # Velocity dispersion
VD_MIN_LYA, VD_INIT_LYA, VD_MAX_LYA = 50,200,700  # Maximum velocity dispersion for Lyman α
GAMMA_MIN, GAMMA_INIT, GAMMA_MAX = -1, 2, 10  # γ parameter for Lyman α
MIN_SNR = 3.0 # Minimum SNR for clipping
WINDOW_MAX = 30 # search radius in A for peak around starting wavelength
MARGIN_EMLINES = 0 # margin in pixel for emission line selection wrt to the spectrum edge
NSTD_RELSIZE = 3.0 # window size relative to FWHM for computation of NSTD
MAXFEV = 100 # maximum iteration by parameter for leastsq


__all__ = ('Linefit', 'fit_lines')


class NoLineError(ValueError):
    """Error raised when there is no line to fit in the spectrum."""

    pass


class Linefit:
    """
    This class implement Emission Line fit
    """
    def __init__(self, vel=(-500,0,500), vdisp=(5,50,300), 
                 velabs=(-500,0,500), vdispabs=(5,50,300),
                 polydegabs=12, polyiterabs=3, polywmask=3.0,
                 vdisp_lya=(50,150,700), gamma_lya=(-1,2,10), 
                 windmax=10, minsnr=3.0,
                 nstd_relsize=3.0,
                 gamma_2lya1 = (-10,-2,0), gamma_2lya2 = (0,2,10),
                 sep_2lya = (80,500,1000),
                 line_ratios = [
                    ("CIII1907", "CIII1909", 0.6, 1.2),
                    ("OII3726", "OII3729", 0.3, 1.5)
                    ],
                 minpars = dict(method='least_square', xtol=1.e-3),
                 mcmcpars = dict(steps=0, nwalkers=0, save_proba=False)
                 ):
        """Initialize line fit parameters and return a Linefit object
          
        Parameters
        ----------
        vel : tuple of floats
          Minimum, init and maximum values of velocity offset in km/s for emission lines (default: -500,0,500).
        velabs: tuple of floats
          Minimum, init and maximum values of rest frame velocity dispersion in km/s for absorption lines (default: 5,80,300). 
        vdisp : tuple of floats
          init and maximum values of rest frame velocity dispersion in km/s for emission lines (default: 5,80,300).
        vdispabs: tuple of floats
          Minimum, init and maximum values of rest frame velocity dispersion in km/s for absorption lines(default: 5,80,300). 
        vdisp_lya : tuple of float
           Minimum, init and maximum values of Lya rest frame velocity dispersion in km/s (default: 50,150,700).
        gamma_lya : tuple of floats
          Minimum, init and maximum values of the skeness parameter for the asymetric gaussain fit (default: -1,2,10).
        gamma_2lya1 : tuple of floats
          Minimum, init and maximum values of the skeness parameter for the left component of the double asymetric gaussain fit (default: -10,-2,0).
        gamma_2lya2 : tuple of floats
          Minimum, init and maximum values of the skeness parameter for the right component of the double asymetric gaussain fit (default: 0,2,10).
        sep_2lya : tuple of floats
          Minimum, init and maximum values of the peak separation in km/s of the double asymetric gaussain fit (default: 50,200,700)        
        windmax : float 
          maximum half size window in A to find peak values around initial wavelength value (default: 10).
        xtol : float
          relative error in the solution for the leastq fitting (default: 1.e-4).
        ftol : float
          relative error in the sum of square for the leastsq fitting (default: 1.e-6).
        maxfev : int
          max number of iterations by parameter for the leastsq fitting (default 50)
        nstd_relsize : float
          relative size (wrt to FWHM) of the wavelength window used for NSTD line estimation (used in bootstrap only), default: 3.0
        minsnr : float
          minimum SNR to display line ID in plots (default 3.0)
        line_ratios : list of tuples
          list of line_ratios, defaulted to [("CIII1907", "CIII1909", 0.6, 1.2), ("OII3726", "OII3729", 1.0, 2.0)] 
        minpars : dictionary
          Parameters to pass to minimize (lmfit) (default dict(method='least_square', xtol=1.e-3) 
      
         
          WIP: polydegabs=12, polyiterabs=3, polywmask=3.0,
              
        Return
        ------
        Linefit object
        
        """    
             
        self.logger = getLogger(__name__)
                
        self.nstd_relsize = nstd_relsize # relative size with respct to FWHM for line NSTD estimate
        
        self.vel = vel # bounds in velocity km/s for emi lines, rest frame
        self.vdisp = vdisp # bounds in velocity dispersion km/s for emi lines, rest frame
        self.velabs = velabs # bounds in velocity km/s for abs lines , rest frame
        self.vdispabs = vdispabs # bounds in velocity dispersion km/s for abs lines, rest frame        
        self.vdisp_lya = vdisp_lya # lya specific bounds in velocity dispersion km/s, rest frame
        self.gamma_lya = gamma_lya # bounds in lya asymmetry
        
        self.polydegabs = polydegabs # polynomial degree for polynomial continuum estimation before absorption line fit
        self.polyiterabs = polyiterabs # maximum iteration for polynomial continuum estimation before absorption line fit
        self.polywmask = polywmask # window half size in A to mask around abs line before continuum fit
        
        self.sep_2lya = sep_2lya
        self.gamma_2lya1 = gamma_2lya1
        self.gamma_2lya2 = gamma_2lya2
        
        self.windmax = windmax # maximum half size window to find peak around initial wavelength value
        self.minsnr = minsnr # minium SNR for writing label of emission line in plot
        
        self.line_ratios = line_ratios # list of line ratios constraints
        
        self.minpars = minpars # dictionary with lmfit minimizatio method and optional parameters
        self.mcmcpars = mcmcpars # dictionary with lmfit EMCEE minimization parameters
                         
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
           line_spec initial spectrum,
           init_linefit spectrum of the starting solution for the line fit,
           line_fit spectrum of the line fit

        """
 
        fit_lws = dict(vel=self.vel, vdisp=self.vdisp, vdisp_lya=self.vdisp_lya, 
                       gamma_lya=self.gamma_lya, minsnr=self.minsnr,
                       nstd_relsize=self.nstd_relsize, sep_2lya=self.sep_2lya, 
                       gamma_2lya1=self.gamma_2lya1, gamma_2lya2=self.gamma_2lya2,
                       )
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
        except:
            unit_data = None
                   
        res = fit_lines(wave=wave, data=data, std=std, redshift=z,
                        unit_wave=u.angstrom, unit_data=unit_data, line_ratios=line_ratios,
                        fit_lws=fit_lws, minpars=self.minpars, mcmcpars=self.mcmcpars, **kwargs)
        
        tab = res['table_spec' ]   
        # convert wave to observed frame and air
        wave = tab['RESTWL']*(1 + z)
        wave = vactoair(wave)
        # add init and fitted spectra on the observed plane
        spfit = line_spec.clone()
        spfit.data = np.interp(line_spec.wave.coord(), wave, tab['LINE_FIT'])
        spfit.data = spfit.data / (1 + z)
        spinit = line_spec.clone()
        spinit.data = np.interp(line_spec.wave.coord(), wave, tab['INIT_FIT'])
        spinit.data = spinit.data / (1 + z)
        
        res['line_spec'] = line_spec    
        res['line_fit'] = spfit
        res['line_initfit'] = spinit
                
        return res  
    
    def absfit(self, spec, z, **kwargs):
        """
        perform an absorption line fit on a mpdaf spectrum
        
        Parameters
        ----------
        line_spec : `mpdaf.obj.Spectrum`
            input spectrum 
        z : float
            initial redshift
        **kwargs : keyword arguments
            Additional arguments passed to the `fit_abs` function.

        Returns
        -------
        res : dictionary
           See `fit_lines`,
           return in addition the following spectrum in the observed frame
           line_spec initial spectrum,
           init_linefit spectrum of the starting solution for the line fit,
           line_fit spectrum of the line fit

        """
        # WIP fit_lws not used ?
        fit_lws = dict(velabs=self.velabs, vdispabs=self.vdispabs, minsnr=self.minsnr,
                       nstd_relsize=self.nstd_relsize)
        
        # remove cont
        deg = self.polydegabs 
        maxiter = self.polyiterabs
        wmask = self.polywmask
        self.logger.debug('Continuum polynomial fit with degre %d itermax %d wmask %.0f A', deg, maxiter, wmask)
        spcont = get_cont(spec, z, deg, maxiter, wmask)
        spabs = spec - spcont
          
        wave = spabs.wave.coord(unit=u.angstrom).copy()
        data = spabs.data   
        if spabs.var is not None:
            std = np.sqrt(spabs.var)
        else:
            std = None
    
        if std is not None:
            bad_points = std == 0
            std[bad_points] = np.inf
    
        try:
            unit_data = u.Unit(spabs.data_header.get("BUNIT", None))
        except:
            unit_data = None
        
        lwargs = kwargs.copy()
        for key in ['major_lines', 'line_ratios', 'fit_all', 'dble_lyafit',
                    'find_lya_vel_offset', 'use_line_ratios']:
            if key in lwargs:
                lwargs.pop(key)
        res = fit_abs(wave=wave, data=data, std=std, redshift=z,
                        unit_wave=u.angstrom, unit_data=unit_data,
                        minpars=self.minpars,
                        **lwargs)          
        
        tab = res['table_spec' ]   
        # convert wave to observed frame and air
        wave = tab['RESTWL']*(1 + z)
        wave = vactoair(wave)
        # add fitted spectra on the observed plane
        spfit = spec.clone()
        spfit.data = np.interp(spec.wave.coord(), wave, tab['ABS_FIT_LSQ'])
        spfit.data = spfit.data / (1 + z)

        res['abs_init'] = spec
        res['abs_cont'] = spcont  
        res['abs_line'] = spfit
        res['abs_fit'] = spfit + spcont
        
                
        return res            
    
    def info(self, res, full_output=False):
        """ Print fit informations 
        
        Parameters
        ----------
        res : dictionary       
              results of `fit`
        """
        res['ztable'].pprint_all(show_unit=False, show_dtype=False)
        
 

            
    def plot(self, ax, res, start=False, iden=True, minsnr=0, line=None, margin=5,
             dplot={'dl':2.0, 'y':0.95, 'size':10}):
        """ plot fit results
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes instance in which to draw the plot       
        res : dictionary 
            results of `fit`           
        start : boolean
            plot initial value before fit
        iden : boolean
            write line label on plot
        minsnr : float
            minimum to write line label 
        margin : float
            margin between left and right wavelength as define in lines table
        start : bool
            plot iniial values before fit
        dplot : dictionary
            parameters to draw line labels
            
            - dl: offset in wavelength
            - y: location in y (0-1)
            - size: font size
            
        
        
        
        """
        plotline(ax, res['line_spec'], res['line_fit'], None, res['line_initfit'], res['lines'], start=start,
                 iden=iden, minsnr=minsnr, line=line, margin=margin, dplot=dplot)

def fit_lines(wave, data, std, redshift, *, unit_wave=None,
                       unit_data=None, vac=False, lines=None, line_ratios=None,
                       major_lines=False, 
                       fit_all=False, lsf=None, trimm_spec=True,
                       find_lya_vel_offset=True, dble_lyafit=False,
                       mcmc_lya=False, mcmc_all=False,
                       fit_lws=None, minpars={}, mcmcpars={}):
    """Fit lines from a set of arrays (wave, data, std) using lmfit.

    This function uses lmfit to perform fit of know lines in
    a spectrum with a given redshift, to compute the flux and flux uncertainty
    of each line covered by the spectrum. It must be used on a continuum
    subtracted spectrum.
    
    The fit are performed by line families, each family is defined by a 
    unique velocity offset and velocity dispersion, plus a list of emission lines.
    All lines are assumed to be gaussian except for lyman-alpha where an
    asymetric gaussian model is used.

    All the emission lines known by `linelist.get_lines` are
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

        line_ratios = [("CIII1907", "CIII1909", 0.6, 1.2), ("OII3726", "OII3729", 1.0, 2.0)]

    That stands for:

        0.6 <= OII3729 / OII3726 <= 1.2
        1.0 <= CIII1909 / CIII1907 <= 2.0


    If the units of the wavelength and data axis are provided, the function
    will try to determine the unit of the computed fluxes.

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
          table must contain the following columns
        
          - LINE: the line name
          - FAMILY: the line family (balmer, forbidden, ism)
          - LBDA_REST: the rest frame wavelength in vacuum (Angstroem)
          - DOUBLET: the central wavelength if this is a multiplet (Angstroem)
          - RESONANT: bool, True if the line is resonant
          - ABS: bool, True if the line can be in absorption
          - EMI: bool, True if the line can be in emission
          - DNAME: the line display name (None if no display)
          
    line_ratios : list of (str, str, float, float) tuples or string
        List on line ratio constraints (line1 name, line2 name, line2/line1
        minimum, line2/line1 maximum.
    major_lines : boolean, optional
        If true, the fit is restricted to the major lines as defined in mpdaf line table (used only when lines is None, )
        default: False
    fit_all : boolean, optional
        if True, use same velocity offset and velocity dispersion for all lines except Lya
        if False, allow different velocity offsets and velocity disperions between balmer,
        forbidden and resonnant lines
        default: false 
    lsf : function, optional
        LSF function LSF to derive FWHM as function of wavelength, otherwise assume no LSF
        default: True
    trimm_spec : boolean, optional
        if True, mask unused wavelengths part
        default : True
    find_lya_vel_offset: boolean, optional
        if True, compute a starting velocity offset for lya on the data [disabled for dble_lyafit]
    dble_lyafit : False
        if True, use a double asymetric gaussian model for the lya line fit
    fit_lws : dictionary with some default and bounds parameters
    minpars : dictionary with minimization method and associate parameters

    Returns
    -------
    result : Dictionary
 
        - lines (see above)
        - ztable (see above)
        - lmfit_{family}  the ResultObject from lmfit for the given family
        - table_spec an astropy table with the following columns
        
            - RESTWL: restframe wavelength
            - FLUX: resframe data value
            - ERR: stddev of FLUX
            - LINEFIT: final fit value
            - INIT_FIT: init value for the full fit
            - {family}_INIT_FIT: initial value for the fit of the given family
            - {family}_FIT_LSQ: LSQ fit of the given family
            
 

    Raises
    ------
    NoLineError: 
        when none of the fitted line can be on the spectrum at the
        given redshift.
    
        
    """
    logger = logging.getLogger(__name__)
    
    logger.debug('Preparing data for fit')
    pdata = prepare_fit_data(wave, data, std, redshift, vac, 
                             lines, major_lines, trimm_spec)
    logger.debug('Initialize fit')
    init_fit(pdata, dble_lyafit, find_lya_vel_offset, lsf, fit_all, line_ratios, fit_lws, mcmc_lya, mcmc_all)
    result = init_res(pdata, mcmc_lya or mcmc_all, save_proba=mcmcpars.get('save_proba',False))
    
    # perform fit
    reslsq = lmfit_fit(minpars, mcmcpars, pdata, verbose=True)   
        
    resfit = save_fit_res(result, pdata, reslsq)
    
    return resfit
        
def prepare_fit_data(wave, data, std, redshift, vac, 
                     lines, major_lines, trimm_spec):
    
    logger = logging.getLogger(__name__)
    
    wave = np.array(wave)
    data = np.array(data)
    std = np.array(std) if std is not None else np.ones_like(data)  
    # convert wavelength in restframe and vacuum, scale flux and std
    wave_rest = airtovac(wave)/(1+redshift)
    data_rest = data*(1+redshift)
    std_rest = std*(1+redshift) 
    
    # mask all points that have a std == 0
    mask = std <= 0
    excluded_lbrange = None
    if np.sum(mask) > 0:
        logger.debug('Masked %d points with std <= 0', np.sum(mask))
        wave_rest, data_rest, std_rest = wave_rest[~mask], data_rest[~mask], std_rest[~mask]               
        if np.sum(mask) > 1:
            excluded_lbrange = find_excluded_lbrange(wave, mask) 
        wave, data, std = wave[~mask], data[~mask], std[~mask] 
        
        
    # Fitting only some lines from reference library.
    if type(lines) is list:
        lines_to_fit = lines
        lines = None
    else:
        lines_to_fit = None    
    
    if lines is None:
        logger.debug("Getting lines from default line table...") 
    else:
        logger.debug("Getting lines from user line table...") 
    main = True if major_lines else None
    lines = get_lines(z=redshift, vac=True, main=main, margin=MARGIN_EMLINES,
                        lbrange=[wave.min(), wave.max()], 
                        exlbrange=excluded_lbrange,
                        emiline=True, restframe=True,
                        user_linetable=lines)
    if lines_to_fit is not None:
        lines = lines[np.in1d(lines['LINE'].tolist(), lines_to_fit)]
        if len(lines) < len(lines_to_fit):
            logger.debug(
                "Some lines are not on the spectrum coverage: %s.",
                ", ".join(set(lines_to_fit) - set(lines['LINE'])))
    lines['LBDA_EXP'] = (1 + redshift) * lines['LBDA_REST']
    if not vac:
        lines['LBDA_EXP'] = vactoair(lines['LBDA_EXP'])        

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
        wave_obs, wave_rest, data_rest, std_rest = wave[mask], wave_rest[mask], data_rest[mask], std_rest[mask]
        logger.debug("%.1f %% of the spectrum is used for fitting.",
                     100 * np.sum(mask) / len(mask)) 
        
    pdata = dict(wave_obs=wave_obs, wave_rest=wave_rest, data_rest=data_rest, std_rest=std_rest,
                 lines=lines, redshift=redshift, vac=vac) 
    
    return pdata

def init_fit(pdata, dble_lyafit, find_lya_vel_offset, lsf, fit_all, line_ratios, fit_lws, mcmc_lya, mcmc_all):
    
    logger = logging.getLogger(__name__)
    
    # get defaut parameters for fit bounds and init values
    init_vel = fit_lws.get('vel',(VEL_MIN,VEL_INIT,VEL_MAX))
    init_vdisp = fit_lws.get('vdisp',(VD_MIN,VD_INIT,VD_MAX))
    init_vdisp_lya = fit_lws.get('vdisp_lya',(VD_MIN_LYA,VD_INIT_LYA,VD_MAX_LYA))
    init_gamma_lya = fit_lws.get('gamma_lya',(GAMMA_MIN,GAMMA_INIT,GAMMA_MAX))
    
    # specific additional parameters for double lya fit
    if dble_lyafit:     
        init_gamma_2lya1 = fit_lws.get('gamma_2lya1')
        init_gamma_2lya2 = fit_lws.get('gamma_2lya2')
        init_sep_2lya = fit_lws.get('sep_2lya')
        find_lya_vel_offset = False 
    
    # get other defaut parameters 
    init_windmax = fit_lws.get('windmax',WINDOW_MAX) # search radius in A for peak around starting wavelength
    init_minsnr = fit_lws.get('minsnr',MIN_SNR) # minimum SNR value for clipping
    nstd_relsize = fit_lws.get('nstd_relsize',NSTD_RELSIZE) # window size relative to FWHM for comutation of NSTD
    pmaxfev = fit_lws.get('maxfev',MAXFEV) # maximum number of iteration by parameter
    
    wave_rest = pdata['wave_rest']
    data_rest = pdata['data_rest']
    lines = pdata['lines']
    redshift = pdata['redshift']
    
    pdata['lsf'] = lsf
    pdata['dble_lyafit'] = dble_lyafit
    pdata['init_minsnr'] = init_minsnr
    pdata['nstd_relsize'] = nstd_relsize
    
    has_lya = 'LYALPHA' in lines['LINE']
    
    if has_lya:
        logger.debug('Init Lya Fit')
        if dble_lyafit:
            logger.debug('Using double asymetric gaussian model')
        sel_lines = lines[lines['LINE']=='LYALPHA']
        # Set input parameters
        params = Parameters()
        if find_lya_vel_offset:
            voff = get_lya_vel_offset(wave_rest, data_rest, box_filter=3)
            init_vel_lya = (init_vel[0]+voff, voff, init_vel[2]+voff)
            logger.debug('Computed Lya init velocity offset: %.2f', voff)
        else:
            init_vel_lya = init_vel
        if dble_lyafit:
            set_dbleasymgaussian_fitpars('lyalpha', params, sel_lines,  
                                 redshift, lsf, init_vel_lya, init_vdisp_lya, 
                                 init_sep_2lya, init_gamma_2lya1, init_gamma_2lya2, 
                                 init_windmax, wave_rest, data_rest)
            family_lines = {'lyalpha': {'fun':'dbleasymgauss', 'lines':['LYALPHA']}}
            maxfev = pmaxfev*7
        else:
            set_asymgaussian_fitpars('lyalpha', params, sel_lines,  
                                     redshift, lsf, init_vel_lya, init_vdisp_lya, init_gamma_lya, init_windmax,
                                     wave_rest, data_rest)
            family_lines = {'lyalpha': {'fun':'asymgauss', 'lines':['LYALPHA']}}
            maxfev = pmaxfev*3
        emcee = True if (mcmc_lya or mcmc_all) else False
        pdata['par_lya'] = dict(params=params, sel_lines=sel_lines, 
                                family_lines=family_lines, maxfev=maxfev, emcee=emcee)
        
    if fit_all:
        logger.debug('Init fit all lines together expect Lya')
        logger.debug('Found %d lines to fit', len(lines) - has_lya)
        sel_lines = lines[lines['LINE'] != 'LYALPHA']
        if len(sel_lines) == 0:
            logger.warning('No lines to fit for all family')           
        else:
            # Set input parameters
            params = Parameters()
            set_gaussian_fitpars('all', params, sel_lines, line_ratios, 
                                 redshift, lsf, init_vel, init_vdisp, init_windmax,
                                 wave_rest, data_rest) 
            family_lines = {'all': {'fun':'gauss', 'lines':sel_lines['LINE']}}
            maxfev = pmaxfev*(2+len(sel_lines))
            emcee = True if mcmc_all else False
            pdata['par_all'] = dict(params=params, sel_lines=sel_lines,
                                family_lines=family_lines, maxfev=maxfev, emcee=emcee)
            
        return
    
    # fitting of families with non resonnant lines
    non_resonant_lines = lines[~lines['RESONANT']]
    families = set(non_resonant_lines['FAMILY'])
    logger.debug('Found %d non resonnant line families to fit', len(families))    
    
    for family in families:
        logger.debug('Init Fit of family %s', family)
        sel_lines = non_resonant_lines[non_resonant_lines['FAMILY']==family]
        logger.debug('Found %d lines to fit', len(sel_lines))
        # Set input parameters
        params = Parameters()
        set_gaussian_fitpars(family, params, sel_lines, line_ratios, 
                             redshift, lsf, init_vel, init_vdisp, init_windmax,
                             wave_rest, data_rest)
        family_lines = {family: {'fun':'gauss', 'lines':sel_lines['LINE']}} 
        maxfev = pmaxfev*(2+len(sel_lines))
        emcee = True if mcmc_all else False
        pdata[f'par_{family}'] = dict(params=params, sel_lines=sel_lines,
                            family_lines=family_lines, maxfev=maxfev, emcee=emcee)        
        
        
    # fitting of families with resonnant lines (except lya, already fitted)
    resonant_lines = lines[lines['RESONANT'] & (lines['LINE']!='LYALPHA')]
    dlines = reorganize_doublets(resonant_lines)    
    logger.debug('Found %d resonnant line families to fit', len(dlines)) 
    for clines in dlines:
        family = clines[0].lower()
        logger.debug('Init fitting of family %s', family)
        ksel = lines['LINE']==clines[0]
        if len(clines) > 1:
            ksel = (lines['LINE']==clines[0]) | (lines['LINE']==clines[1])
        sel_lines = lines[ksel]           
        logger.debug('Found %d lines to fit', len(sel_lines)) 
        # Set input parameters
        params = Parameters()
        set_gaussian_fitpars(family, params, sel_lines, line_ratios, 
                             redshift, lsf, init_vel, init_vdisp, init_windmax,
                             wave_rest, data_rest)
        family_lines = {family: {'fun':'gauss', 'lines':sel_lines['LINE']}}
        maxfev = pmaxfev*(2+len(sel_lines))
        emcee = True if mcmc_all else False
        pdata[f'par_{family}'] = dict(params=params, sel_lines=sel_lines,
                            family_lines=family_lines, maxfev=maxfev, emcee=emcee)      

    
def init_res(pdata, mcmc, save_proba=False):
    
    logger = logging.getLogger(__name__) 
    
    # initialize result tables
    # set tablines for results by lines
    tablines = Table()
    colnames = ['LBDA_REST','VEL','VEL_ERR','Z','Z_ERR','Z_INIT','VDISP','VDISP_ERR',
                    'FLUX','FLUX_ERR','SNR','SKEW','SKEW_ERR',
                    'LBDA_OBS','PEAK_OBS','LBDA_LEFT','LBDA_RIGHT','FWHM_OBS','NSTD', 
                    'LBDA_LNSTD','LBDA_RNSTD'] 
    for colname in colnames:
        tablines.add_column(MaskedColumn(name=colname, dtype=np.float, mask=True))    
    if mcmc:
        k = list(tablines.columns).index('VEL_ERR')
        tablines.add_column(MaskedColumn(name='VEL_RTAU', dtype=np.float, mask=True), index=k+1)
        k = list(tablines.columns).index('VDISP_ERR')
        tablines.add_column(MaskedColumn(name='VDISP_RTAU', dtype=np.float, mask=True), index=k+1)
        k = list(tablines.columns).index('FLUX_ERR')
        tablines.add_column(MaskedColumn(name='FLUX_RTAU', dtype=np.float, mask=True), index=k+1)
        k = list(tablines.columns).index('SKEW_ERR')
        tablines.add_column(MaskedColumn(name='SKEW_RTAU', dtype=np.float, mask=True), index=k+1)
        colnames += ['VEL_RTAU','VDISP_RTAU','FLUX_RTAU','SKEW_RTAU']
        if save_proba:
            for key in ['VEL','VDISP','FLUX','SKEW','Z']:
                k = list(tablines.columns).index(key+'_ERR')
                tablines.add_column(MaskedColumn(name=key+'_MAX99', dtype=np.float, mask=True), index=k+1)
                tablines.add_column(MaskedColumn(name=key+'_MAX95', dtype=np.float, mask=True), index=k+1) 
                tablines.add_column(MaskedColumn(name=key+'_MIN95', dtype=np.float, mask=True), index=k+1)
                tablines.add_column(MaskedColumn(name=key+'_MIN99', dtype=np.float, mask=True), index=k+1) 
                colnames += [key+'_MAX99', key+'_MAX95', key+'_MIN95', key+'_MIN99']
    tablines.add_column(MaskedColumn(name='BLEND', dtype=np.int, mask=True))
    tablines.add_column(MaskedColumn(name='FAMILY', dtype='U20', mask=True), index=0)
    tablines.add_column(MaskedColumn(name='LINE', dtype='U20', mask=True), index=1)
    tablines.add_column(MaskedColumn(name='DNAME', dtype='U20', mask=True), index=3)
    k = list(tablines.columns).index('LINE')
    tablines.add_column(MaskedColumn(name='ISBLEND', dtype=np.bool, mask=True), index=k+1)     
    if pdata['lsf'] is not None:
        k = list(tablines.columns).index('VDISP')
        tablines.add_column(MaskedColumn(name='VDINST', dtype=np.float, mask=True), index=k)
        colnames.append('VDINST')
    if pdata['dble_lyafit']:
        k = list(tablines.columns).index('VDISP')
        tablines.add_column(MaskedColumn(name='SEP', dtype=np.float, mask=True), index=k)
        colnames.append('SEP') 
        k = list(tablines.columns).index('VDISP')
        tablines.add_column(MaskedColumn(name='SEP_ERR', dtype=np.float, mask=True), index=k)
        colnames.append('SEP_ERR')
        if mcmc:
            k = list(tablines.columns).index('VDISP')
            tablines.add_column(MaskedColumn(name='SEP_RTAU', dtype=np.float, mask=True), index=k)
            colnames.append('SEP_RTAU') 
            if save_proba:
                for key in ['SEP']:
                    k = list(tablines.columns).index(key+'_ERR')
                    tablines.add_column(MaskedColumn(name=key+'_MAX99', dtype=np.float, mask=True), index=k+1)
                    tablines.add_column(MaskedColumn(name=key+'_MAX95', dtype=np.float, mask=True), index=k+1) 
                    tablines.add_column(MaskedColumn(name=key+'_MIN95', dtype=np.float, mask=True), index=k+1)
                    tablines.add_column(MaskedColumn(name=key+'_MIN99', dtype=np.float, mask=True), index=k+1) 
                    colnames += [key+'_MAX99', key+'_MAX95', key+'_MIN99', key+'_MIN95']            
    for colname in colnames:
        tablines[colname].format = '.2f'
    tablines['Z'].format = '.5f'
    tablines['Z_INIT'].format = '.5f'
    if 'Z_MAX99' in tablines.columns:
        for c in ['Z_MAX99','Z_MAX95','Z_MIN95','Z_MIN99']:
            tablines[c].format = '.5f'
    tablines['Z_ERR'].format = '.2e'
    #set ztable for global results by family 
    ztab = Table()
    ztab.add_column(MaskedColumn(name='FAMILY', dtype='U20', mask=True))
    colnames =  ['VEL','VEL_ERR','Z','Z_ERR','Z_INIT','VDISP','VDISP_ERR','SNRMAX','SNRSUM','SNRSUM_CLIPPED']
    for colname in colnames:
        ztab.add_column(MaskedColumn(name=colname, dtype=np.float, mask=True))
    for colname in colnames:
        ztab[colname].format = '.2f'
    ztab.add_column(MaskedColumn(name='LINE', dtype='U20', mask=True), index=8)
    for colname in ['NL','NL_CLIPPED','NFEV','STATUS']:
            ztab.add_column(MaskedColumn(name=colname, dtype=np.int, mask=True)) 
    ztab.add_column(MaskedColumn(name='RCHI2', dtype=np.float, mask=True), index=14)
    ztab.add_column(MaskedColumn(name='METHOD', dtype='U25', mask=True))
    if mcmc:
        ztab.add_column(MaskedColumn(name='RCHAIN', dtype=np.float, mask=True)) 
        ztab.add_column(MaskedColumn(name='NSTEPS', dtype=np.int, mask=True)) 
        ztab['RCHAIN'].format = '.2f'
    ztab['RCHI2'].format = '.2f'   
    ztab['Z'].format = '.5f'
    ztab['Z_ERR'].format = '.2e'
    ztab['Z_INIT'].format = '.5f'
    ztab.add_index('FAMILY')
    
    # set tablespec for spectrum fit
    tabspec = Table(data=[pdata['wave_rest'],pdata['data_rest'],pdata['std_rest']], 
                    names=['RESTWL','FLUX','ERR'])
    tabspec['LINE_FIT'] = tabspec['FLUX']*0
    tabspec['INIT_FIT'] = tabspec['FLUX']*0
    
    return dict(tabspec=tabspec, tablines=tablines, ztab=ztab)
        

def lmfit_fit(minpars, mcmcpars, pdata, verbose=True):
    
    logger = logging.getLogger(__name__)
    
    reslsq = {}
    
    # Perform minimization 
    parlist = [e for e in pdata.keys() if e[0:4]=='par_']
    for par in parlist:
        if verbose:
            logger.debug(f'Fitting of Line Family: {par[4:]}')
        args =  (pdata['wave_rest'], pdata['data_rest'], pdata['std_rest'],  
                 pdata[par]['family_lines'], pdata['redshift'], pdata['lsf'])
        minner = Minimizer(residuals, pdata[par]['params'], fcn_args=args) 
        if verbose:
            logger.debug('Lmfit fitting: %s',minpars)
        result = minner.minimize(**minpars)
        if verbose:
            logger.debug('%s after %d iterations, reached minimum = %.3f and redChi2 = %.3f',result.message,
                         result.nfev,result.chisqr,result.redchi)
        # MCMC 
        if pdata[par]['emcee']:
            nwalkers = mcmcpars.pop('nwalkers',0)
            steps = mcmcpars.pop('steps',0)
            save_proba = mcmcpars.pop('save_proba',False)
            emceepars = {**dict(method='emcee', is_weighted=True, progress=verbose), **mcmcpars}
            emceepars['nwalkers'] = 10*result.nvarys if nwalkers==0 else nwalkers
            if steps == 0:
                emceepars['steps'] = 15000 if 'lyalpha_LYALPHA_dbleasymgauss_l0' in pdata[par]['params'].keys() else 10000
            else:
                emceepars['steps'] = steps
            if verbose:
                logger.debug('Emcee fitting: %s',emceepars)                  
            minner = Minimizer(residuals, result.params, fcn_args=args) 
            # run EMCEE
            resmcmc = minner.minimize(**emceepars)
            # Check if autocorr integ time has been successful
            if not hasattr(resmcmc,'acor'):
                # run an estimate of the autocorr integration time
                resmcmc.acor = emcee.autocorr.integrated_time(resmcmc.chain, tol=0)
            # save the parameter values for the highest proba  
            highest_prob = np.argmax(resmcmc.lnprob)
            hp_loc = np.unravel_index(highest_prob, resmcmc.lnprob.shape)
            mle_soln = resmcmc.chain[hp_loc]
            i = 0
            for p in resmcmc.params:
                if resmcmc.params[p].vary:
                    resmcmc.params[p].median_value = copy.copy(resmcmc.params[p].value)
                    resmcmc.params[p].init_stderr = result.params[p].stderr
                    resmcmc.params[p].value = mle_soln[i] 
                    resmcmc.params[p].acor = resmcmc.acor[i]
                    resmcmc.params[p].acor_ratio = resmcmc.chain.shape[0]/(resmcmc.acor[i]*50)
                    if save_proba:
                        quantiles = np.percentile(resmcmc.flatchain[p],[5,95,1,99]) 
                        resmcmc.params[p].min_p95 = quantiles[0]
                        resmcmc.params[p].max_p95 = quantiles[1] 
                        resmcmc.params[p].min_p99 = quantiles[2]
                        resmcmc.params[p].max_p99 = quantiles[3]                                              
                    i += 1
            resmcmc.mean_acceptance_fraction = np.mean(resmcmc.acceptance_fraction)
            resmcmc.max_acor = max(resmcmc.acor)
            resmcmc.chain_size_ratio = resmcmc.chain.shape[0]/(resmcmc.max_acor*50)
            resmcmc.status = 1 if resmcmc.chain_size_ratio > 1 else 0
            if verbose:
                logger.debug('status %d after %d fcn eval, chain size ratio = %.1f max autocorr time = %.1f mean acceptance fraction = %.2f, reached minimum = %.3f and redChi2 = %.3f',
                             resmcmc.status,resmcmc.nfev,resmcmc.chain_size_ratio,resmcmc.max_acor,resmcmc.mean_acceptance_fraction,resmcmc.chisqr,resmcmc.redchi)
            reslsq[f'{par[4:]}'] = resmcmc
        else:
            reslsq[f'{par[4:]}'] = result
        bestfit = model(reslsq[f'{par[4:]}'].params, pdata['wave_rest'], 
                            pdata[par]['family_lines'], pdata['redshift'], pdata['lsf']) 
        reslsq[f'{par[4:]}'].bestfit = bestfit         
        
        
    return reslsq

def compute_bestfit(reslsq, pdata):
    bestfits = []
    for key,res in reslsq.items():
        bestfits.append(model(res.params, pdata['wave_rest'], 
                    pdata['par_'+key]['family_lines'], pdata['redshift'], pdata['lsf']))
    bestfit = np.sum(bestfits, axis=0)
    return bestfit

def save_fit_res(result, pdata, reslsq):
    
    logger = logging.getLogger(__name__)
    
    # save fit init and result to tabspec
    tabspec = result['tabspec']
    tablines = result['tablines']
    ztab = result['ztab']
    
    wave_rest = pdata['wave_rest']   
    redshift = pdata['redshift']
    lsf = pdata['lsf']
    
    init_minsnr = pdata['init_minsnr']
    vac = pdata['vac']
    
    resfit = {}
    
    for key in reslsq.keys():
        parname = 'par_'+key
        logger.debug('Saving %s results to tablines and ztab', key)
        family_lines = pdata[parname]['family_lines'] 
        sel_lines = pdata[parname]['sel_lines']
        init_params = pdata[parname]['params']
        
        tabspec[f'{key.upper()}_INIT_FIT'] = model(init_params, wave_rest, family_lines, redshift, lsf)
        tabspec[f'{key.upper()}_FIT_LSQ'] = model(reslsq[key].params, wave_rest, family_lines, redshift, lsf) 
        tabspec['LINE_FIT'] = tabspec['LINE_FIT'] + tabspec[f'{key.upper()}_FIT_LSQ'] 
        tabspec['INIT_FIT'] = tabspec['INIT_FIT'] + tabspec[f'{key.upper()}_INIT_FIT']
        
        add_result_to_tablines(reslsq[key], tablines, redshift, sel_lines, lsf, vac)
        add_line_stat_to_table(reslsq[key], pdata, sel_lines, tablines)  
        resfit[f'lmfit_{key}'] = reslsq[key]
        
    add_blend_to_table(tablines) 
    add_result_to_ztab(reslsq, tablines, ztab, init_minsnr)
   
    resfit['table_spec'] = tabspec 
    
    tablines.sort('LBDA_REST')
    resfit['lines'] = tablines
    
      
    resfit['ztable'] = ztab      
    
    return resfit

           
def add_line_stat_to_table(reslsq, pdata, sel_lines, tablines):
    kfactor = pdata['nstd_relsize']
    for row in sel_lines:
        line = row['LINE']
        if (line=='LYALPHA') and (pdata['dble_lyafit']):
            lmask = (tablines['LINE']=='LYALPHA1') | (tablines['LINE']=='LYALPHA2')
        elif row['DOUBLET'] > 0:
            dlines = sel_lines[np.abs(sel_lines['DOUBLET']-row['DOUBLET'])<0.01]['LINE']
            lmask = np.in1d([str(e) for e in tablines['LINE']],[str(e) for e in dlines])
        else:
            lmask = tablines['LINE']==line
        selrows = tablines[lmask]
        wave = pdata['wave_obs']
        mask = np.full_like(wave, False, dtype=bool)  # Points to keep
        left = 1.e20
        right = 0
        for selrow in selrows:     
            l0 = selrow['LBDA_OBS']
            l1 = selrow['LBDA_LEFT']
            l2 = selrow['LBDA_RIGHT']
            l1 = l0 - kfactor*(l0 - l1)
            l2 = l0 + kfactor*(l2 - l0)
            mask[(wave>=l1) & (wave<=l2)] = True
            left = min(l1,left)
            right = max(l2,right)
        bestfit = reslsq.bestfit[mask]
        data = pdata['data_rest'][mask]
        norm = np.sum(bestfit)
        nstd = np.log10(np.std((data-bestfit)/norm)) if abs(norm) > 1.e-10 else 100.
        tablines['NSTD'][lmask] = nstd
        tablines['LBDA_LNSTD'][lmask] = left
        tablines['LBDA_RNSTD'][lmask] = right
        
def add_blend_to_table(tablines):
    tab = tablines[tablines['BLEND']>0]
    if len(tab) == 0:
        return
    blist = np.unique(tab['BLEND'])
    for b in blist:
        d = {}
        # name for the line blend
        stab = tab[tab['BLEND']==b]
        name = stab['LINE'][0]
        if (name == 'LYALPHA1') or (name == 'LYALPHA2'):
            d['LINE'] = line = 'LYALPHAb'
        else:
            for k,c in enumerate(name):
                if c.isdigit():
                    break 
            if k < len(name)-1:
                d['LINE'] = line = f"{name[:k]}{b}b"
            else:
                d['LINE'] = line = f"{name}b"
        d['BLEND'] = b
        d['ISBLEND'] = True
        d['FAMILY'] = family = stab['FAMILY'][0]
        t = stab[stab['DNAME'] != 'None']
        if len(t) == 0:
            d['DNAME'] = 'None'
        else:
            d['DNAME'] = t['DNAME'][0]
        # FLUX 
        d['FLUX'] = np.sum(stab['FLUX'])
        # we compute an estimate of SNR by sq root of sum of SNR**2 
        d['SNR'] = np.sqrt(np.sum(stab['SNR']**2))
        d['FLUX_ERR'] = np.abs(d['FLUX']/d['SNR']) 
        # VDISP, FWHM
        d['VDISP'] = np.average(stab['VDISP'])
        d['VDISP_ERR'] = np.average(stab['VDISP_ERR'])
        d['FWHM_OBS'] = np.average(stab['FWHM_OBS'])
        # LBDA
        d['LBDA_OBS'] = np.average(stab['LBDA_OBS'])
        d['LBDA_LEFT'] = np.min(stab['LBDA_LEFT'])
        d['LBDA_RIGHT'] = np.max(stab['LBDA_RIGHT'])
        d['LBDA_REST'] = np.average(stab['LBDA_REST'])
        d['PEAK_OBS'] = np.max(stab['PEAK_OBS'])
        # Z, VEL
        d['Z'] = np.mean(stab['Z'])
        d['Z_ERR'] = np.mean(stab['Z_ERR'])
        d['Z_INIT'] = np.mean(stab['Z_INIT'])
        d['VEL'] = np.mean(stab['VEL'])
        d['VEL_ERR'] = np.mean(stab['VEL_ERR'])
        if line == 'LYALPHAb':
            kmax = np.argmax(stab['SKEW'])
            d['SKEW'] = stab['SKEW'][kmax]
            d['SKEW_ERR'] = stab['SKEW_ERR'][kmax]
        upsert_ltable(tablines, d, family, line)       
        
    return
        

def reorganize_doublets(lines):
    dlines = [[e['LINE']] for e in lines[lines['DOUBLET']==0]]
    doublets = lines[lines['DOUBLET']>0]
    ndoublets = set(doublets['DOUBLET'])
    for dlbda in ndoublets:
        slines = doublets[np.abs(doublets['DOUBLET']-dlbda) < 0.01]
        dlines.append([e['LINE'] for e in slines])
    return dlines
    



def add_result_to_tablines(result, tablines, zinit, inputlines, lsf, vac):
    """ add results to the table, if row exist it is updated"""
    par = result.params
    families = [key.split('_')[1] for key in par.keys() if key.split('_')[0]=='dv']
    for family in families: 
        dv = par[f"dv_{family}"].value
        dv_err = par[f"dv_{family}"].stderr
        dv_rtau = par[f"dv_{family}"].acor_ratio if hasattr(par[f"dv_{family}"], 'acor_ratio') else None
        lines = [key.split('_')[1] for key in par.keys() if (key.split('_')[0]==family) and (key.split('_')[3]=='l0')]
        for line in lines:
            dname = inputlines[inputlines['LINE']==line]['DNAME'][0]
            blend = inputlines[inputlines['LINE']==line]['DOUBLET'][0]
            keys = [key for key in par.keys() if key.split('_')[1]==line]
            fun = keys[0].split('_')[2]
            if fun == 'dbleasymgauss':
                ndv = dv
                vdisp1 = par[f"vdisp1_{family}"].value 
                vdisp1_err = par[f"vdisp1_{family}"].stderr
                vdisp1_rtau = par[f"vdisp1_{family}"].acor_ratio if hasattr(par[f"vdisp1_{family}"], 'acor_ratio') else None
                vdisp2 = par[f"vdisp2_{family}"].value 
                vdisp2_err = par[f"vdisp2_{family}"].stderr 
                vdisp2_rtau = par[f"vdisp2_{family}"].acor_ratio if hasattr(par[f"vdisp2_{family}"], 'acor_ratio') else None
                vdisp = 0.5*(vdisp1 + vdisp2)
                sep = par[f"{family}_{line}_{fun}_sep"].value
                sep_err = par[f"{family}_{line}_{fun}_sep"].stderr 
                sep_rtau = par[f"{family}_{line}_{fun}_sep"].acor_ratio if hasattr(par[f"{family}_{line}_{fun}_sep"], 'acor_ratio') else None
                flux1 = par[f"{family}_{line}_{fun}_flux1"].value
                flux1_err = par[f"{family}_{line}_{fun}_flux1"].stderr
                flux1_rtau = par[f"{family}_{line}_{fun}_flux1"].acor_ratio if hasattr(par[f"{family}_{line}_{fun}_flux1"], 'acor_ratio') else None
                flux2 = par[f"{family}_{line}_{fun}_flux2"].value
                flux2_err = par[f"{family}_{line}_{fun}_flux2"].stderr   
                flux2_rtau = par[f"{family}_{line}_{fun}_flux2"].acor_ratio if hasattr(par[f"{family}_{line}_{fun}_flux2"], 'acor_ratio') else None
                l0 = par[f"{family}_{line}_{fun}_l0"].value
                l1 = l0*(1 + dv/C)
                l1obs = l1*(1+zinit)
                # the redshift is given at the mid-point of the 2 asymetric gaussian
                z = l1obs/l0 - 1 # compute redshift in vacuum 
                lvals1 = {'LBDA_REST':l0, 'LBDA_OBS':l1obs, 'FLUX':flux1, 
                         'DNAME':dname,  'VDISP':vdisp1, 'BLEND':int(l0+0.5),
                         'Z_INIT':zinit, 'ISBLEND':False,
                         'VEL':dv, 'Z':z, 'SEP':sep
                         } 
                lvals2 = {'LBDA_REST':l0, 'LBDA_OBS':l1obs, 'FLUX':flux2, 
                          'DNAME':dname,  'VDISP':vdisp2, 'BLEND':int(l0+0.5),
                          'Z_INIT':zinit, 'ISBLEND':False,
                          'VEL':dv, 'Z':z, 'SEP':sep
                          } 
                if sep_err is not None:
                    lvals1['SEP_ERR'] = sep_err
                    lvals2['SEP_ERR'] = sep_err
                if sep_rtau is not None:
                    lvals1['SEP_RTAU'] = sep_rtau
                    lvals2['SEP_RTAU'] = sep_rtau
                    if 'SEP_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals1['SEP_'+suff] = getattr(par[f"{family}_{line}_{fun}_sep"], attr)
                            lvals2['SEP_'+suff] = getattr(par[f"{family}_{line}_{fun}_sep"], attr)
                if dv_err is not None:
                    lvals1['VEL_ERR'] = dv_err
                    z_err = (1+zinit)*dv_err/C 
                    lvals1['Z_ERR'] = z_err
                    lvals2['VEL_ERR'] = dv_err
                    lvals2['Z_ERR'] = z_err 
                else:
                    z_err = None 
                if dv_rtau is not None:
                    lvals1['VEL_RTAU'] = dv_rtau
                    lvals2['VEL_RTAU'] = dv_rtau 
                    if 'VEL_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals1['VEL_'+suff] = getattr(par[f"dv_{family}"], attr)
                            lvals2['VEL_'+suff] = getattr(par[f"dv_{family}"], attr) 
                            lvals1['Z_'+suff] = zinit + (1+zinit)*lvals1['VEL_'+suff]/C 
                            lvals2['Z_'+suff] = zinit + (1+zinit)*lvals2['VEL_'+suff]/C
                if vdisp1_err is not None:
                    lvals1['VDISP_ERR'] = vdisp1_err
                    lvals2['VDISP_ERR'] = vdisp2_err 
                    vdisp_err = np.sqrt(vdisp1_err**2+vdisp2_err**2)
                else:
                    vdisp_err = None
                if vdisp1_rtau is not None:
                    lvals1['VDISP_RTAU'] = vdisp1_rtau
                    lvals2['VDISP_RTAU'] = vdisp2_rtau 
                    if 'VDISP_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals1['VDISP_'+suff] = getattr(par[f"vdisp1_{family}"], attr)
                            lvals2['VDISP_'+suff] = getattr(par[f"vdisp2_{family}"], attr)                     
                if flux1_err is not None:
                    lvals1['FLUX_ERR'] = flux1_err 
                    lvals1['SNR'] = abs(flux1)/flux1_err 
                    lvals2['FLUX_ERR'] = flux2_err 
                    lvals2['SNR'] = abs(flux2)/flux2_err  
                if flux1_rtau is not None:
                    lvals1['FLUX_RTAU'] = flux1_rtau
                    lvals2['FLUX_RTAU'] = flux2_rtau  
                    if 'FLUX_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals1['FLUX_'+suff] = getattr(par[f"{family}_{line}_{fun}_flux1"], attr)
                            lvals2['FLUX_'+suff] = getattr(par[f"{family}_{line}_{fun}_flux2"], attr)                       
                if lsf is not None:
                    lvals1['VDINST'] = complsf(lsf, l1obs, kms=True) 
                    lvals2['VDINST'] = complsf(lsf, l1obs, kms=True)  
                skew1 = par[f"{family}_{line}_{fun}_asym1"].value
                lvals1['SKEW'] = skew1
                skew2 = par[f"{family}_{line}_{fun}_asym2"].value
                lvals2['SKEW'] = skew2                
                skew1_err = par[f"{family}_{line}_{fun}_asym1"].stderr 
                skew2_err = par[f"{family}_{line}_{fun}_asym2"].stderr 
                skew1_rtau = par[f"{family}_{line}_{fun}_asym1"].acor_ratio if hasattr(par[f"{family}_{line}_{fun}_asym1"], 'acor_ratio') else None
                skew2_rtau = par[f"{family}_{line}_{fun}_asym2"].acor_ratio if hasattr(par[f"{family}_{line}_{fun}_asym2"], 'acor_ratio') else None
                if skew1_err is not None:
                    lvals1['SKEW_ERR'] = skew1_err
                    lvals2['SKEW_ERR'] = skew2_err 
                if skew1_rtau is not None:
                    lvals1['SKEW_RTAU'] = skew1_rtau
                    lvals2['SKEW_RTAU'] = skew2_rtau  
                    if 'SKEW_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals1['SKEW_'+suff] = getattr(par[f"{family}_{line}_{fun}_asym1"], attr)
                            lvals2['SKEW_'+suff] = getattr(par[f"{family}_{line}_{fun}_asym2"], attr)                         
                # find the line peak location in rest frame
                swave_rest = np.linspace(l0-50,l0+50,1000)
                l1c = l0*(1+(dv-0.5*sep)/C)
                sigma1 = get_sigma(vdisp1, l1c, z, lsf, restframe=True) 
                peak1 = flux1/(SQRT2PI*sigma1)
                vmodel_rest = asymgauss(peak1, l1c, sigma1, skew1, swave_rest)
                kmax = np.argmax(vmodel_rest)    
                l1 = swave_rest[kmax]
                left_rest,right_rest = rest_fwhm_asymgauss(swave_rest, vmodel_rest)
                l1obs = l1*(1+zinit)
                l1left = left_rest*(1+zinit)
                l1right = right_rest*(1+zinit)
                if not vac:
                    l1obs = vactoair(l1obs)
                    l1left = vactoair(l1left)
                    l1right = vactoair(l1right) 
                # compute the peak value and convert it to observed frame    
                lvals1['PEAK_OBS'] = np.max(vmodel_rest)/(1+zinit)
                # save peak position in observed frame
                lvals1['LBDA_OBS'] = l1obs
                lvals1['LBDA_LEFT'] = l1left
                lvals1['LBDA_RIGHT'] = l1right                    
                lvals1['FWHM_OBS'] = l1right - l1left
                # 2nd peak
                l2c = l0*(1+(dv+0.5*sep)/C)
                sigma2 = get_sigma(vdisp2, l1c, z, lsf, restframe=True) 
                peak2 = flux2/(SQRT2PI*sigma1)
                vmodel_rest = asymgauss(peak2, l2c, sigma2, skew2, swave_rest)                
                kmax = np.argmax(vmodel_rest)    
                l1 = swave_rest[kmax]
                left_rest,right_rest = rest_fwhm_asymgauss(swave_rest, vmodel_rest)
                l1obs = l1*(1+zinit)
                l1left = left_rest*(1+zinit)
                l1right = right_rest*(1+zinit)
                if not vac:
                    l1obs = vactoair(l1obs)
                    l1left = vactoair(l1left)
                    l1right = vactoair(l1right) 
                # compute the peak value and convert it to observed frame    
                lvals2['PEAK_OBS'] = np.max(vmodel_rest)/(1+zinit)
                # save peak position in observed frame
                lvals2['LBDA_OBS'] = l1obs
                lvals2['LBDA_LEFT'] = l1left
                lvals2['LBDA_RIGHT'] = l1right                    
                lvals2['FWHM_OBS'] = l1right - l1left                
                # update line table
                upsert_ltable(tablines, lvals1, family, 'LYALPHA1') 
                upsert_ltable(tablines, lvals2, family, 'LYALPHA2')  
            elif fun == 'asymgauss':
                vdisp = par[f"vdisp_{family}"].value 
                vdisp_err = par[f"vdisp_{family}"].stderr
                vdisp_rtau = par[f"vdisp_{family}"].acor_ratio if hasattr(par[f"vdisp_{family}"], 'acor_ratio') else None
                flux = par[f"{family}_{line}_{fun}_flux"].value
                flux_err = par[f"{family}_{line}_{fun}_flux"].stderr
                flux_rtau = par[f"{family}_{line}_{fun}_flux"].acor_ratio if hasattr(par[f"{family}_{line}_{fun}_flux"], 'acor_ratio') else None
                l0 = par[f"{family}_{line}_{fun}_l0"].value
                l1 = l0*(1 + dv/C)
                l1obs = l1*(1+zinit)
                z = l1obs/l0 - 1 # compute redshift in vacuum 
                if not vac:
                    l1obs = vactoair(l1obs)
                lvals = {'LBDA_REST':l0, 'LBDA_OBS':l1obs, 'FLUX':flux, 
                         'DNAME':dname,  'VDISP':vdisp, 'BLEND':blend,
                         'Z_INIT':zinit, 'ISBLEND':False,
                         }
                if dv_rtau is not None:
                    lvals['VEL_RTAU'] = dv_rtau 
                    if 'VEL_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals['VEL_'+suff] = getattr(par[f"dv_{family}"], attr)
                            lvals['Z_'+suff] = zinit + (1+zinit)*lvals['VEL_'+suff]/C                                       
                if vdisp_rtau is not None:
                    lvals['VDISP_RTAU'] = vdisp_rtau
                    if 'VDISP_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals['VDISP_'+suff] = getattr(par[f"vdisp_{family}"], attr)                        
                if flux_rtau is not None:
                    lvals['FLUX_RTAU'] = flux_rtau 
                    if 'FLUX_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals['FLUX_'+suff] = getattr(par[f"{family}_{line}_{fun}_flux"], attr)                    
                if vdisp_err is not None:
                    lvals['VDISP_ERR'] = vdisp_err 
                if flux_err is not None:
                    lvals['FLUX_ERR'] = flux_err 
                    lvals['SNR'] = abs(flux)/flux_err 
                if lsf is not None:
                    lvals['VDINST'] = complsf(lsf, l1obs, kms=True)       
                skew = par[f"{family}_{line}_{fun}_asym"].value
                skew_rtau = par[f"{family}_{line}_{fun}_asym"].acor_ratio if hasattr(par[f"{family}_{line}_{fun}_asym"], 'acor_ratio') else None
                lvals['SKEW'] = skew
                skew_err = par[f"{family}_{line}_{fun}_asym"].stderr 
                if skew_err is not None:
                    lvals['SKEW_ERR'] = skew_err
                if skew_rtau is not None:
                    lvals['SKEW_RTAU'] = skew_rtau
                    if 'SKEW_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals['SKEW_'+suff] = getattr(par[f"{family}_{line}_{fun}_asym"], attr)                    
                # find the line peak location in rest frame
                swave_rest = np.linspace(l0-50,l0+50,1000)
                vmodel_rest = model_asymgauss(zinit, lsf, l0, flux, skew, vdisp, dv, swave_rest)
                kmax = np.argmax(vmodel_rest)    
                l1 = swave_rest[kmax]
                left_rest,right_rest = rest_fwhm_asymgauss(swave_rest, vmodel_rest)
                # this position is used for redshift and dv
                ndv = C*(l1-l0)/l0 # offset in km/s
                l1obs = l1*(1+zinit)
                l1left = left_rest*(1+zinit)
                l1right = right_rest*(1+zinit)
                z = l1obs/l0 - 1 # compute redshift in vacuum
                if not vac:
                    l1obs = vactoair(l1obs)
                    l1left = vactoair(l1left)
                    l1right = vactoair(l1right)
                # save in table
                lvals['VEL'] = ndv
                dv = ndv
                if dv_err is not None:
                    lvals['VEL_ERR'] = dv_err
                    z_err = dv_err*(1+zinit)/C
                    lvals['Z_ERR'] = z_err 
                else:
                    z_err = None
                lvals['Z'] = z  
                # compute the peak value and convert it to observed frame    
                lvals['PEAK_OBS'] = np.max(vmodel_rest)/(1+zinit)
                # save peak position in observed frame
                lvals['LBDA_OBS'] = l1obs
                lvals['LBDA_LEFT'] = l1left
                lvals['LBDA_RIGHT'] = l1right                    
                lvals['FWHM_OBS'] = l1right - l1left
                # update line table
                upsert_ltable(tablines, lvals, family, line)
            elif fun == 'gauss': 
                ndv = dv
                vdisp = par[f"vdisp_{family}"].value 
                vdisp_err = par[f"vdisp_{family}"].stderr
                vdisp_rtau = par[f"vdisp_{family}"].acor_ratio if hasattr(par[f"vdisp_{family}"], 'acor_ratio') else None
                flux = par[f"{family}_{line}_{fun}_flux"].value
                flux_err = par[f"{family}_{line}_{fun}_flux"].stderr
                flux_rtau = par[f"{family}_{line}_{fun}_flux"].acor_ratio if hasattr(par[f"{family}_{line}_{fun}_flux"], 'acor_ratio') else None
                l0 = par[f"{family}_{line}_{fun}_l0"].value
                l1 = l0*(1 + dv/C)
                l1obs = l1*(1+zinit)
                z = l1obs/l0 - 1 # compute redshift in vacuum 
                if not vac:
                    l1obs = vactoair(l1obs)
                lvals = {'LBDA_REST':l0, 'LBDA_OBS':l1obs, 'FLUX':flux, 
                         'DNAME':dname,  'VDISP':vdisp, 'BLEND':blend,
                         'Z_INIT':zinit, 'ISBLEND':False, 
                         } 
                if dv_rtau is not None:
                    lvals['VEL_RTAU'] = dv_rtau
                    if 'VEL_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals['VEL_'+suff] = getattr(par[f"dv_{family}"], attr)
                            lvals['Z_'+suff] = zinit + (1+zinit)*lvals['VEL_'+suff]/C                         
                if vdisp_rtau is not None:
                    lvals['VDISP_RTAU'] = vdisp_rtau
                    if 'VDISP_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals['VDISP_'+suff] = getattr(par[f"vdisp_{family}"], attr)                       
                if flux_rtau is not None:
                    lvals['FLUX_RTAU'] = flux_rtau  
                    if 'FLUX_MAX99' in tablines.columns:
                        for suff,attr in [['MIN95','min_p95'],['MIN99','min_p99'],['MAX95','max_p95'],['MAX99','max_p99']]:
                            lvals['FLUX_'+suff] = getattr(par[f"{family}_{line}_{fun}_flux"], attr)                                  
                if vdisp_err is not None:
                    lvals['VDISP_ERR'] = vdisp_err 
                if (flux_err is not None) and (flux_err > 0):
                    lvals['FLUX_ERR'] = flux_err 
                    lvals['SNR'] = abs(flux)/flux_err 
                if lsf is not None:
                    lvals['VDINST'] = complsf(lsf, l1obs, kms=True)
                sigma = get_sigma(vdisp, l1obs, z, lsf, restframe=False)
                fwhm = 2.355*sigma
                lvals.update({'FWHM_OBS':fwhm, 'LBDA_LEFT':l1obs-0.5*fwhm, 'LBDA_RIGHT':l1obs+0.5*fwhm, 
                         'PEAK_OBS':flux/(SQRT2PI*sigma), 'VEL':dv, 'Z':z}) 
                if dv_err is not None:
                    lvals['VEL_ERR'] = dv_err
                    z_err = dv_err*(1+zinit)/C
                    lvals['Z_ERR'] = z_err
                else:
                    z_err = None
                    
                # update line table
                upsert_ltable(tablines, lvals, family, line)
            else:
                raise ValueError('fun %s unknown'%(fun))
    

def add_result_to_ztab(reslsq, tablines, ztab, snr_min):
    families = np.unique(tablines['FAMILY'])
    lines = tablines[tablines['ISBLEND'] | (tablines['BLEND']==0)]
    for fam in families:
        result = reslsq[fam if fam != 'lyalpha' else 'lya']
        cat = lines[lines['FAMILY']==fam]
        tcat = cat[cat['SNR']>0]
        status = -99
        for key in ['status','ier']: #get output status
            if hasattr(result, key):
                status = getattr(result, key)
                break          
        if len(tcat) == 0:
            d = dict(FAMILY=fam, VEL=cat['VEL'][0], VEL_ERR=cat['VEL_ERR'][0],
                 Z=cat['Z'][0], Z_ERR=cat['Z_ERR'][0], Z_INIT=cat['Z_INIT'][0],
                 VDISP=cat['VDISP'][0], VDISP_ERR=cat['VDISP_ERR'][0],
                 SNRMAX=0, SNRSUM_CLIPPED=0, NL_CLIPPED=0, LINE='None',
                 NL=len(cat), RCHI2=result.redchi, NFEV=result.nfev, 
                 STATUS=status, METHOD=result.method)
        else:
            kmax = np.argmax(cat['SNR'])
            scat = tcat[(tcat['SNR']>snr_min) ]
            d = dict(FAMILY=fam, VEL=cat['VEL'][0], VEL_ERR=cat['VEL_ERR'][0],
                     Z=cat['Z'][0], Z_ERR=cat['Z_ERR'][0], Z_INIT=cat['Z_INIT'][0],
                     VDISP=cat['VDISP'][0], VDISP_ERR=cat['VDISP_ERR'][0],
                     SNRMAX=cat['SNR'][kmax], LINE=cat['LINE'][kmax], 
                     SNRSUM=np.abs(np.sum(tcat['FLUX']))/np.sqrt(np.sum(tcat['FLUX_ERR']**2)),
                     SNRSUM_CLIPPED = np.abs(np.sum(scat['FLUX']))/np.sqrt(np.sum(scat['FLUX_ERR']**2)) if len(scat) > 0 else 0,
                     NL=len(cat), NL_CLIPPED=len(scat), RCHI2=result.redchi, NFEV=result.nfev, 
                     STATUS=status, METHOD=result.method)  
        if hasattr(result,'chain_size_ratio'):
            d['RCHAIN'] = result.chain_size_ratio
        if hasattr(result,'chain'): 
            d['NSTEPS'] = result.chain.shape[0]
        upsert_ztable(ztab, d, fam)
        
    ztab.sort('SNRSUM')
    ztab = ztab[::-1]           

             
def upsert_ztable(tab, vals, family):
    if family in tab['FAMILY']:
        # update
        row = tab.loc[family]
        for k,v in vals.items():
            row[k] = v
    else:
        # add line
        vals['FAMILY'] = family
        tab.add_row(vals)

def upsert_ltable(tab, vals, family, line):
    if (family in tab['FAMILY']) and (line in tab['LINE']):
        # update
        ksel = (tab['FAMILY']==family) & (tab['LINE']==line)
        for k,v in vals.items():
            tab[k][ksel] = v
    else:
        # add line
        vals['FAMILY'] = family
        vals['LINE'] = line
        tab.add_row(vals)             
        
def add_gaussian_par(params, family_name, name, l0, z, vdisp, lsf, wind_max, wave, data, absline=False):
    ksel = np.abs(wave-l0) < wind_max
    params.add(f"{family_name}_{name}_gauss_l0", value=l0, vary=False)  
    sigma = get_sigma(vdisp, l0, z, lsf, restframe=True)                      
    if absline:
        vmax = data[ksel].min()
        flux = SQRT2PI*sigma*vmax        
        params.add(f"{family_name}_{name}_gauss_flux", value=flux, max=0)
    else:
        vmax = data[ksel].max()
        flux = SQRT2PI*sigma*vmax
        params.add(f"{family_name}_{name}_gauss_flux", value=flux, min=0)
    
def add_asymgauss_par(params, family_name, name, l0, z, vdisp, lsf, wind_max, gamma, wave, data):   
    ksel = np.abs(wave-l0) < wind_max 
    params.add(f"{family_name}_{name}_asymgauss_l0", value=l0, vary=False) 
    vmax = data[ksel].max()
    sigma = get_sigma(vdisp, l0, z, lsf, restframe=True)                  
    flux = SQRT2PI*sigma*vmax
    params.add(f"{family_name}_{name}_asymgauss_flux", value=flux, min=0)
    if np.all(np.isclose(gamma, gamma[0])):
        params.add(f"{family_name}_{name}_asymgauss_asym", value=gamma[1], vary=False)
    else:
        params.add(f"{family_name}_{name}_asymgauss_asym", value=gamma[1], min=gamma[0], max=gamma[2])

def add_dbleasymgauss_par(params, family_name, name, l0, z, vdisp, lsf, wind_max, sep, gamma1, gamma2, wave, data):     
    ksel = np.abs(wave-l0) < wind_max
    params.add(f"{family_name}_{name}_dbleasymgauss_l0", value=l0, vary=False)
    vmax = data[ksel].max()
    sigma = get_sigma(vdisp, l0, z, lsf, restframe=True)                  
    flux1 = SQRT2PI*sigma*vmax*0.3
    flux2 = SQRT2PI*sigma*vmax
    params.add(f"{family_name}_{name}_dbleasymgauss_sep", value=sep[1],
               min=sep[0], max=sep[2])
    params.add(f"{family_name}_{name}_dbleasymgauss_flux1", value=flux1, min=0)
    params.add(f"{family_name}_{name}_dbleasymgauss_flux2", value=flux2, min=0)
    params.add(f"{family_name}_{name}_dbleasymgauss_asym1", value=gamma1[1], 
               min=gamma1[0], max=gamma1[2])
    params.add(f"{family_name}_{name}_dbleasymgauss_asym2", value=gamma2[1], 
               min=gamma2[0], max=gamma2[2])
     
    
def get_lya_vel_offset(wave, data, box_filter=3):
    l0 = 1215.67
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
    
def set_gaussian_fitpars(family_name, params, lines, line_ratios, z, lsf, init_vel, 
                         init_dv, windmax, wave, data,
                         absline=False):
    logger = logging.getLogger(__name__)
    # add velocity and velocity dispersion fit parameters
    if np.all(np.isclose(init_vel, init_vel[0])):
        params.add(f'dv_{family_name}', value=init_vel[1], vary=False)
    else:
        params.add(f'dv_{family_name}', value=init_vel[1], min=init_vel[0], max=init_vel[2])
    if np.all(np.isclose(init_dv, init_dv[0])):
        params.add(f'vdisp_{family_name}', value=init_dv[1], vary=False) 
    else:
        params.add(f'vdisp_{family_name}', value=init_dv[1], min=init_dv[0], max=init_dv[2]) 
    # we use gaussian parameters
    nc = 0
    vdisp = init_dv[1]
    for line in lines:      
        name = line['LINE']
        l0 = line['LBDA_REST']
        add_gaussian_par(params, family_name, name, l0, z, vdisp, lsf, windmax, wave, data, absline)
    logger.debug('added %d gaussian to the fit', len(lines))
    if line_ratios is not None:
        # add line ratios bounds
        dlines = lines[lines['DOUBLET']>0]
        if len(dlines) > 0:
            nc += 1
            add_line_ratio(family_name, params, line_ratios, dlines) 
    if nc > 0: logger.debug('added %d doublet constrains to the fit', nc)
    
def set_asymgaussian_fitpars(family_name, params, lines, z, lsf, init_vel, 
                         init_dv, init_gamma, windmax, wave, data):
    logger = logging.getLogger(__name__)
    # add velocity and velocity dispersion fit parameters
    if np.all(np.isclose(init_vel, init_vel[0])):
        params.add(f'dv_{family_name}', value=init_vel[1], vary=False)
    else:
        params.add(f'dv_{family_name}', value=init_vel[1], min=init_vel[0], max=init_vel[2])    
    if np.all(np.isclose(init_dv, init_dv[0])):
        params.add(f'vdisp_{family_name}', value=init_dv[1], vary=False) 
    else:
        params.add(f'vdisp_{family_name}', value=init_dv[1], min=init_dv[0], max=init_dv[2]) 
    # we use asymetric gaussian parameters
    nc = 0
    vdisp = init_dv[1]
    for line in lines:      
        name = line['LINE']
        l0 = line['LBDA_REST']
        add_asymgauss_par(params, family_name, name, l0, z, vdisp, lsf, 
                          windmax, init_gamma, wave, data)
    logger.debug('added %d asymetric gaussian to the fit', len(lines))
    
def set_dbleasymgaussian_fitpars(family_name, params, lines, z, lsf, init_vel, 
                         init_dv, init_sep, init_gamma1, init_gamma2, 
                         windmax, wave, data):
    logger = logging.getLogger(__name__)
    # add velocity and velocity dispersion fit parameters
    params.add(f'dv_{family_name}', value=init_vel[1], min=init_vel[0], max=init_vel[2])
    params.add(f'vdisp1_{family_name}', value=init_dv[1], min=init_dv[0], max=init_dv[2]) 
    params.add(f'vdisp2_{family_name}', value=init_dv[1], min=init_dv[0], max=init_dv[2]) 
    # we use asymetric gaussian parameters
    nc = 0
    vdisp = init_dv[1]
    for line in lines:      
        name = line['LINE']
        l0 = line['LBDA_REST']
        add_dbleasymgauss_par(params, family_name, name, l0, z, vdisp, lsf, windmax, 
                              init_sep, init_gamma1, init_gamma2, wave, data)
    logger.debug('added %d double asymetric gaussian to the fit', len(lines))

          
    
def add_line_ratio(family, params, line_ratios, dlines):
    for line1,line2,ratio_min,ratio_max in line_ratios:
        if (line1 in dlines['LINE']) and (line2 in dlines['LINE']):
            params.add("%s_to_%s_factor" % (line1, line2), min=ratio_min,
                       max=ratio_max, value=0.5*(ratio_min+ratio_max))
            params['%s_%s_gauss_flux' % (family, line2)].expr = (
                "%s_%s_gauss_flux * %s_to_%s_factor" % (family, line1, line1, line2)
            ) 
            
def get_sigma(vdisp, l0, z, lsf=None, restframe=True):
    sigma = vdisp*l0/C
    if lsf is not None:
        if restframe:
            l0obs = l0*(1+z)
            siginst = complsf(lsf, l0obs)/(1+z)
        else:
            siginst = complsf(lsf, l0)
        sigma = np.sqrt(sigma**2+siginst**2) 
    return sigma


def model(params, wave, lines, z, lsf=None):
    """ wave is rest frame wavelengths """
    logger = logging.getLogger(__name__)
    model = 0
    for name,ldict in lines.items():       
        dv = params[f"dv_{name}"].value
        for line in ldict['lines']:
            if ldict['fun']=='gauss':
                vdisp = params[f"vdisp_{name}"].value
                flux = params[f"{name}_{line}_gauss_flux"].value
                l0 = params[f"{name}_{line}_gauss_l0"].value
                model += model_gauss(z, lsf, l0, flux, vdisp, dv, wave)
            elif ldict['fun']=='asymgauss':
                vdisp = params[f"vdisp_{name}"].value
                flux = params[f"{name}_{line}_asymgauss_flux"].value
                l0 = params[f"{name}_{line}_asymgauss_l0"].value
                beta = params[f"{name}_{line}_asymgauss_asym"].value  
                model += model_asymgauss(z, lsf, l0, flux, beta, vdisp, dv, wave)
            elif ldict['fun']=='dbleasymgauss':
                vdisp1 = params[f"vdisp1_{name}"].value
                vdisp2 = params[f"vdisp2_{name}"].value
                flux1 = params[f"{name}_{line}_dbleasymgauss_flux1"].value
                flux2 = params[f"{name}_{line}_dbleasymgauss_flux2"].value
                l0 = params[f"{name}_{line}_dbleasymgauss_l0"].value
                beta1 = params[f"{name}_{line}_dbleasymgauss_asym1"].value 
                beta2 = params[f"{name}_{line}_dbleasymgauss_asym2"].value
                sep = params[f"{name}_{line}_dbleasymgauss_sep"].value
                model += model_dbleasymgauss(z, lsf, l0, sep, flux1, flux2, 
                                             beta1, beta2, vdisp1, vdisp2, 
                                             dv, wave)             
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

def model_dbleasymgauss(z, lsf, l0, sep, flux1, flux2, 
                        beta1, beta2, vdisp1, vdisp2, dv, wave):
    l1 = l0*(1+(dv-0.5*sep)/C)
    sigma1 = get_sigma(vdisp1, l1, z, lsf, restframe=True)  
    l2 = l0*(1+(dv+0.5*sep)/C)
    sigma2 = get_sigma(vdisp2, l2, z, lsf, restframe=True)
    peak1 = flux1/(SQRT2PI*sigma1)
    peak2 = flux2/(SQRT2PI*sigma2)
    model = dbleasymgauss(peak1, peak2, l1, l2, sigma1, sigma2, beta1, beta2, wave)
    return model

def model_gauss(z, lsf, l0, flux, vdisp, dv, wave):
    l1 = l0*(1+dv/C)
    sigma = get_sigma(vdisp, l1, z, lsf, restframe=True)               
    peak = flux/(SQRT2PI*sigma)
    model = gauss(peak, l1, sigma, wave)
    return model

def complsf(lsfmodel, wave, kms=False):
    # compute estimation of LSF in A
    fwhm = lsfmodel(wave)
    sigma = fwhm/2.355
    if kms:
        sigma = sigma*C/wave
    return sigma

    
def residuals(params, wave, data, std, lines, z, lsf=None):
    vmodel = model(params, wave, lines, z, lsf)
    res = (vmodel - data)/std 
    return res
    
def gauss(peak, l0, sigma, wave):
    g = peak*np.exp(-(wave-l0)**2/(2*sigma**2))
    return g

def dbleasymgauss(peak1, peak2, l1, l2, sigma1, sigma2, 
                  gamma1, gamma2, wave):
    f1 = asymgauss(peak1, l1, sigma1, gamma1, wave)
    f2 = asymgauss(peak2, l2, sigma2, gamma2, wave)
    return f1+f2

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
        l1 = row['LBDA_LEFT'] - margin
        l2 = row['LBDA_RIGHT'] + margin
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
    
    
def fit_abs(wave, data, std, redshift, *, unit_wave=None,
            unit_data=None, vac=False, lines=None, 
            lsf=None, trimm_spec=True, mcmc_all=False,
            fit_lws={}, minpars={}, mcmcpars={}):    
    
    logger = logging.getLogger(__name__)
    
    logger.debug('Preparing data for fit')     
    pdata = prepare_absfit_data(wave, data, std, redshift, vac,
                             lines, trimm_spec)
    logger.debug('Initialize fit')
    init_absfit(pdata, lsf, fit_lws, mcmc=mcmc_all)
    result = init_res(pdata, mcmc_all, save_proba=mcmcpars.get('save_proba',False))

    # perform lsq fit
    reslsq = lmfit_fit(minpars, pdata, verbose=True)       
        
    resfit = save_fit_res(result, pdata, reslsq)
    
    return resfit    

def prepare_absfit_data(wave, data, std, redshift, vac, 
                     lines, trimm_spec):
    
    logger = logging.getLogger(__name__)
    
    wave = np.array(wave)
    data = np.array(data)
    std = np.array(std) if std is not None else np.ones_like(data)  
    # convert wavelength in restframe and vacuum, scale flux and std
    wave_rest = airtovac(wave)/(1+redshift)
    data_rest = data*(1+redshift)
    std_rest = std*(1+redshift) 
    
    # mask all points that have a std == 0
    mask = std <= 0
    excluded_lbrange = None
    if np.sum(mask) > 0:
        logger.debug('Masked %d points with std <= 0', np.sum(mask))
        wave_rest, data_rest, std_rest = wave_rest[~mask], data_rest[~mask], std_rest[~mask] 
        excluded_lbrange = find_excluded_lbrange(wave, mask)
        wave, data, std = wave[~mask], data[~mask], std[~mask]         
        
    # Fitting only some lines from reference library.
    if type(lines) is list:
        lines_to_fit = lines
        lines = None
    else:
        lines_to_fit = None    
    
    if lines is None:
        logger.debug("Getting lines from default line table...") 
    else:
        logger.debug("Getting lines from user line table...") 
    lines = get_lines(z=redshift, vac=True, margin=0,
                        lbrange=[wave.min(), wave.max()], 
                        exlbrange=excluded_lbrange,
                        absline=True, restframe=True,
                        user_linetable=lines)
    if lines_to_fit is not None:
        lines = lines[np.in1d(lines['LINE'].tolist(), lines_to_fit)]
        if len(lines) < len(lines_to_fit):
            logger.debug(
                "Some lines are not on the spectrum coverage: %s.",
                ", ".join(set(lines_to_fit) - set(lines['LINE'])))
    lines['LBDA_EXP'] = (1 + redshift) * lines['LBDA_REST']
    if not vac:
        lines['LBDA_EXP'] = vactoair(lines['LBDA_EXP'])        


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
            vd_max = VD_MAX
            wave_min = line_wave * (1 + VEL_MIN / C)
            wave_min -= 3 * wave_min * vd_max / C
            wave_max = line_wave * (1 + VEL_MAX / C)
            wave_max += 3 * wave_max * vd_max / C
            mask[(wave_rest >= wave_min) & (wave_rest <= wave_max)] = True
        wave_obs, wave_rest, data_rest, std_rest = wave[mask], wave_rest[mask], data_rest[mask], std_rest[mask]
        logger.debug("%.1f %% of the spectrum is used for fitting.",
                     100 * np.sum(mask) / len(mask)) 
        
    pdata = dict(wave_obs=wave_obs, wave_rest=wave_rest, data_rest=data_rest, std_rest=std_rest,
                 lines=lines, redshift=redshift, vac=vac) 
    
    return pdata

def init_absfit(pdata, lsf, fit_lws, mcmc=False):
    
    logger = logging.getLogger(__name__)
    
    # get defaut parameters for fit bounds and init values
    init_velabs = fit_lws.get('velabs', (VEL_MIN,VEL_INIT,VEL_MAX))
    init_vdispabs = fit_lws.get('vdispabs', (VD_MIN,VD_INIT,VD_MAX))
    
    # get other defaut parameters 
    init_windmax = fit_lws.get('windmax',WINDOW_MAX) # search radius in A for peak around starting wavelength
    init_minsnr = fit_lws.get('minsnr',MIN_SNR) # minimum SNR value for clipping
    nstd_relsize = fit_lws.get('nstd_relsize',NSTD_RELSIZE) # window size relative to FWHM for comutation of NSTD
    pmaxfev = fit_lws.get('maxfev',MAXFEV) # maximum number of iteration by parameter

    
    wave_rest = pdata['wave_rest']
    data_rest = pdata['data_rest']
    lines = pdata['lines']
    redshift = pdata['redshift']
    
    pdata['lsf'] = lsf
    pdata['init_minsnr'] = init_minsnr
    pdata['nstd_relsize'] = nstd_relsize
    pdata['dble_lyafit'] = False

    logger.debug('Init absorption lines fit')
    logger.debug('Found %d lines to fit', len(lines))
    # Set input parameters
    line_ratios = None
    params = Parameters()
    set_gaussian_fitpars('abs', params, lines, line_ratios, 
                         redshift, lsf, init_velabs, init_vdispabs, init_windmax,
                         wave_rest, data_rest, absline=True) 
    family_lines = {'abs': {'fun':'gauss', 'lines':lines['LINE']}}
    maxfev = pmaxfev*(2+len(lines))
    pdata['par_abs'] = dict(params=params, sel_lines=lines,
                        family_lines=family_lines, maxfev=maxfev, emcee=mcmc)
    
def get_cont(spec, z, deg, maxiter, width):
    sp = spec.copy()
    if width > 0:          
        wave = sp.wave.coord()
        lines = get_lines(z=z, vac=False, margin=0,
                            lbrange=[wave.min(), wave.max()], 
                            absline=True, restframe=False) 
        for line in lines['LBDA_OBS']:
            sp.mask_region(lmin=line-width, lmax=line+width)   
    spcont = sp.poly_spec(deg, maxiter=maxiter)
    return spcont


def find_excluded_lbrange(wave, mask, minwave=3):
    data = np.arange(len(wave))[mask].tolist()
    excluded = []
    for group in mit.consecutive_groups(data):
        group = list(group)
        if len(group) > minwave:
            excluded.append([wave[group[0]], wave[group[-1]]])        
    return excluded
