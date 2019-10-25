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
warnings.filterwarnings("ignore", message="Initial state is not linearly independent and it will not allow a full exploration of parameter space")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

C = constants.c.to(u.km / u.s).value
SQRT2PI = np.sqrt(2*np.pi)

# Parameters used in the fitting
VEL_MIN, VEL_INIT, VEL_MAX = -500, 0, 500  # Velocity
VD_MIN, VD_INIT, VD_MAX = 10, 50, 300  # Velocity dispersion
VD_MAX_LYA = 700  # Maximum velocity dispersion for Lyman α
GAMMA_MIN, GAMMA_INIT, GAMMA_MAX = -1, 0, 10  # γ parameter for Lyman α
MIN_SNR = 3.0 # Minimum SNR for clipping
WINDOW_MAX = 30 # search radius in A for peak around starting wavelength
MARGIN_EMLINES = 0 # margin in pixel for emission line selection wrt to the spectrum edge

family_names = ['Abs','balmer','forbidden','resonnant']

__all__ = ('Linefit', 'fit_lines')


class NoLineError(ValueError):
    """Error raised when there is no line to fit in the spectrum."""

    pass


class Linefit:
    """
    This class implement Emission Line fit
    """
    def __init__(self, vel=(-500,0,500), vdisp=(5,50,300), 
                 vdisp_lya_max=700, gamma_lya=(-1,0,10), 
                 delta_vel=100, delta_vdisp=50, delta_gamma=0.5,
                 windmax=10, xtol=1.e-4, ftol=1.e-6, maxfev=1000, minsnr=3.0,
                 steps=1000, nwalkers=0, burn=20, seed=None, progress=False,
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
        delta_vel : float
          Maximum excursion of Velocity Offset with respect to the LSQ solution 
          used for EMCEE fit (default is to keep the same constrains as LSQ)
        delta_vdisp : float
          Maximum excursion of Velocity dispersion with respect to the LSQ solution 
          used for EMCEE fit (default is to keep the same constrains as LSQ)
        delta_gamma : float
          Maximum excursion of gamma with respect to the LSQ solution 
          used for EMCEE fit (default is to keep the same constrains as LSQ)         
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
        progress : bool
          if True display progress bar during EMCEE computation (default: False)
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
        self.progress = progress # enable emcee progress bar
        
        self.vel = vel # bounds in velocity km/s, rest frame
        self.vdisp = vdisp # bounds in velocity dispersion km/s, rest frame
        self.vdisp_lya_max = vdisp_lya_max # maximum lya velocity dispersion km/s, rest frame
        self.gamma = gamma_lya # bounds in lya asymmetry
        
        self.delta_vel = delta_vel # max excursion in EMCEE fit wrt LSQ solution
        self.delta_vdisp = delta_vdisp # max excursion in EMCEE fit wrt LSQ solution
        self.delta_gamma = delta_gamma # max excursion in EMCEE fit wrt LSQ solution
        
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
           line_spec initial spectrum,
           init_linefit spectrum of the starting solution for the line fit,
           line_fit spectrum of the line fit

        """
 
        lsq_kws = dict(maxfev=self.maxfev, xtol=self.xtol, ftol=self.ftol)
        mcmc_kws = dict(steps=self.steps, nwalkers=self.nwalkers, burn=self.burn, seed=self.seed, progress=self.progress)
        fit_lws = dict(vel=self.vel, vdisp=self.vdisp, vdisp_lya_max=self.vdisp_lya_max, 
                       gamma=self.gamma, minsnr=self.minsnr,
                       delta_vel=self.delta_vel, delta_vdisp=self.delta_vdisp, 
                       delta_gamma=self.delta_gamma
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
        except ValueError:
            unit_data = None
                   
        res = fit_lines(wave=wave, data=data, std=std, redshift=z,
                        unit_wave=u.angstrom, unit_data=unit_data, line_ratios=line_ratios,
                        lsq_kws=lsq_kws, mcmc_kws=mcmc_kws, fit_lws=fit_lws,
                        **kwargs)
        
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
                       major_lines=False, emcee=False,
                       fit_all=False, lsf=True, trimm_spec=True,
                       find_lya_vel_offset=True,
                       lsq_kws=None, mcmc_kws=None, fit_lws=None):
    """Fit lines from a set of arrays (wave, data, std) using lmfit.

    This function uses lmfit to perform fit of know lines in
    a spectrum with a given redshift, to compute the flux and flux uncertainty
    of each line covered by the spectrum. It must be used on a continuum
    subtracted spectrum.
    
    The fit are performed by line families, each family is defined by a 
    unique velocity offset and velocity dispersion, plus a list of emission lines.
    All lines are assumed to be gaussian except for lyman-alpha where an
    asymetric gaussian model is used.

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
          
    line_ratios : list of (str, str, float, float) tuples or string
        List on line ratio constraints (line1 name, line2 name, line2/line1
        minimum, line2/line1 maximum.
    major_lines : boolean, optional
        If true, the fit is restricted to the major lines as defined in mpdaf line table (used only when lines is None, )
        default: False
    emcee : boolean, optional
        if true, errors and best fit is estimated with MCMC starting from the leastsq solution
        default: False
    fit_all : boolean, optional
        if True, use same velocity offset and velocity dispersion for all lines except Lya
        if False, allow different velocity offsets and velocity disperions between balmer,
        forbidden and resonnant lines
        default: false 
    lsf : boolean, optional
        if True, use LSF estimate to derive instrumental PSF, otherwise assume no LSF
        default: True
    trimm_spec : boolean, optional
        if True, mask unused wavelengths part
        default : True
    find_lya_vel_offset: boolean, optional
        if True, compute a starting velocity offset for lya on the data
    lsq_kws : dictionary with leasq parameters (see scipy.optimize.leastsq)
    mcmc_kws : dictionary with MCMC parameters (see emcee)
    fit_lws : dictionary with some default and bounds parameters

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
            - {family}_FIT_EMCEE: EMCEE fit of the given family
            
       
            
 
        

    Raises
    ------
    NoLineError: 
        when none of the fitted line can be on the spectrum at the
        given redshift.
        
    """
    logger = logging.getLogger(__name__)
    
    resfit = {} # result dictionary

    wave = np.array(wave)
    data = np.array(data)
    std = np.array(std) if std is not None else np.ones_like(data)
    
    # get defaut parameters for fit bounds and init values
    init_vel = fit_lws.get('vel',(VEL_MIN,VEL_INIT,VEL_MAX))
    init_vdisp = fit_lws.get('vdisp',(VD_MIN,VD_INIT,VD_MAX))
    init_vdisp_lya_max = fit_lws.get('vdisp_lya_max',VD_MAX_LYA)
    init_gamma_lya = fit_lws.get('gamma_lya',(GAMMA_MIN,GAMMA_INIT,GAMMA_MAX))
    
    # get default relative bounds with respect to LSQ solution for 2nd EMCEE fit
    if emcee:
        init_delta_vel = fit_lws.get('delta_vel',None)
        init_delta_vdisp = fit_lws.get('delta_vdisp',None)
        init_delta_gamma = fit_lws.get('delta_gamma',None)
    
    # get other defaut parameters 
    init_windmax = fit_lws.get('windmax',WINDOW_MAX) # search radius in A for peak around starting wavelength
    init_minsnr = fit_lws.get('minsnr',MIN_SNR) # minimum SNR value for clipping
    
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
        
    
    # initialize result tables
    # set tablines for results by lines
    tablines = Table()
    colnames = ['LBDA_REST','VEL','VEL_ERR','Z','Z_ERR','Z_INIT','VDISP','VDISP_ERR',
                    'FLUX','FLUX_ERR','SNR','SKEW','SKEW_ERR','LBDA_OBS',
                    'PEAK_OBS','LBDA_LEFT','LBDA_RIGHT','FWHM_OBS', 'RCHI2'] 
    for colname in colnames:
        tablines.add_column(MaskedColumn(name=colname, dtype=np.float, mask=True))
    tablines.add_column(MaskedColumn(name='FAMILY', dtype='U20', mask=True), index=0)
    tablines.add_column(MaskedColumn(name='LINE', dtype='U20', mask=True), index=1)
    tablines.add_column(MaskedColumn(name='DNAME', dtype='U20', mask=True), index=3)
    if lsf:
        tablines.add_column(MaskedColumn(name='VDINST', dtype=np.float, mask=True), index=11)
        colnames.append('VDINST')
    for colname in colnames:
        tablines[colname].format = '.2f'
    tablines['Z'].format = '.5f'
    tablines['Z_INIT'].format = '.5f'
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
    for colname in ['NL','NL_CLIPPED', 'NFEV']:
            ztab.add_column(MaskedColumn(name=colname, dtype=np.int, mask=True)) 
    ztab.add_column(MaskedColumn(name='RCHI2', dtype=np.float, mask=True))
    ztab['RCHI2'].format = '.2f'
    ztab['Z'].format = '.5f'
    ztab['Z_ERR'].format = '.2e'
    ztab['Z_INIT'].format = '.5f'
    ztab.add_index('FAMILY')
    
    # set tablespec for spectrum fit
    tabspec = Table(data=[wave_rest,data_rest,std_rest], 
                    names=['RESTWL','FLUX','ERR'])
    tabspec['LINE_FIT'] = tabspec['FLUX']*0
    tabspec['INIT_FIT'] = tabspec['FLUX']*0

    
    # The fitting is done with lmfit. 
    # The model is a sum of Gaussian (or skewed Gaussian), one per line.
    #
    # Fitting strategy
    # if fit_all, all lines except Lya are fitted together with the same dv and vdisp
    # else all families are fitted independantly in the following order
    # 1. all balmer lines
    # 2. all forbidden lines
    # 3. lya
    # 4+ all other resonnant lines (done separately)ha
    
    has_lya = 'LYALPHA' in lines['LINE']
    
    if has_lya:
        logger.debug('LSQ Fitting of Lya')
        sel_lines = lines[lines['LINE']=='LYALPHA']
        # Set input parameters
        params = Parameters()
        init_vdisp_lya = (init_vdisp[0], init_vdisp[1], init_vdisp_lya_max)
        if find_lya_vel_offset:
            voff = get_lya_vel_offset(wave_rest, data_rest, box_filter=3)
            init_vel_lya = (init_vel[0]+voff, voff, init_vel[2]+voff)
            logger.debug('Computed Lya init velocity offset: %.2f', voff)
        else:
            init_vel_lya = init_vel
        set_asymgaussian_fitpars('lyalpha', params, sel_lines,  
                             redshift, lsf, init_vel_lya, init_vdisp_lya, init_gamma_lya, init_windmax,
                             wave_rest, data_rest)
        family_lines = {'lyalpha': {'fun':'asymgauss', 'lines':['LYALPHA']}}
        # Perform LSQ fit    
        minner = Minimizer(residuals, params, fcn_args=(wave_rest, data_rest, std_rest, family_lines, redshift, lsf))            
        logger.debug('Leastsq fitting with ftol: %.0e xtol: %.0e maxfev: %d',lsq_kws['ftol'],lsq_kws['xtol'],lsq_kws['maxfev'])
        result_lya = minner.minimize(**lsq_kws)
        logger.debug('%s after %d iterations, redChi2 = %.3f',result_lya.message,result_lya.nfev,result_lya.redchi)
        # save fit init and result to tabspec
        tabspec['LYA_INIT_FIT'] = model(params, wave_rest, family_lines, redshift, lsf)
        tabspec['LYA_FIT_LSQ'] = model(result_lya.params, wave_rest, family_lines, redshift, lsf) 
        # Perform MCMC fit
        if emcee: 
            update_bounds(result_lya, init_delta_vel, init_delta_vdisp, init_delta_gamma)
            mdict = set_nwakers(mcmc_kws, result_lya)                
            logger.debug('Error estimation using EMCEE with nsteps: %d nwalkers: %d burn: %d',mdict['steps'],mdict['nwalkers'],mdict['burn'])
            result_lya = minner.emcee(params=result_lya.params, is_weighted=True, float_behavior='chi2', **mdict)
            logger.debug('End EMCEE after %d iterations, redChi2 = %.3f',result_lya.nfev,result_lya.redchi) 
            # save fit emcee result to tabspec
            tabspec['LYA_FIT_EMCEE'] = model(result_lya.params, wave_rest, family_lines, redshift, lsf)
        # save results
        logger.debug('Saving results to tablines and ztab')
        add_result_to_tables(result_lya, tablines, ztab, redshift, sel_lines, lsf, init_minsnr, vac)  
        resfit['lmfit_lyalpha'] = result_lya         
        resfit['lines'] = tablines
        resfit['ztable'] = ztab  
        if 'LYA_FIT_EMCEE' in tabspec.colnames:
            tabspec['LINE_FIT'] = tabspec['LINE_FIT'] + tabspec['LYA_FIT_EMCEE'] 
        else:
            tabspec['LINE_FIT'] = tabspec['LINE_FIT'] + tabspec['LYA_FIT_LSQ'] 
        tabspec['INIT_FIT'] = tabspec['INIT_FIT'] + tabspec['LYA_INIT_FIT']
        resfit['table_spec'] = tabspec
    
    if fit_all:
        logger.debug('Performing fitting of all expect Lya lines together')
        logger.debug('LSQ Fitting of %d lines', len(lines) - has_lya)
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
            # Perform LSQ fit    
            minner = Minimizer(residuals, params, fcn_args=(wave_rest, data_rest, std_rest, family_lines, redshift, lsf))            
            logger.debug('Leastsq fitting with ftol: %.0e xtol: %.0e maxfev: %d',lsq_kws['ftol'],lsq_kws['xtol'],lsq_kws['maxfev'])
            result_all = minner.minimize(**lsq_kws)
            logger.debug('%s after %d iterations, redChi2 = %.3f',result_all.message,result_all.nfev,result_all.redchi)
            # save fit init and result to tabspec
            tabspec['ALL_INIT_FIT'] = model(params, wave_rest, family_lines, redshift, lsf)
            tabspec['ALL_FIT_LSQ'] = model(result_all.params, wave_rest, family_lines, redshift, lsf)            
            # Perform MCMC fit
            if emcee:
                update_bounds(result_all, init_delta_vel, init_delta_vdisp, init_delta_gamma)
                mdict = set_nwakers(mcmc_kws, result_all)
                logger.debug('Error estimation using EMCEE with nsteps: %d nwalkers: %d burn: %d',mdict['steps'],mdict['nwalkers'],mdict['burn'])
                result_lya = minner.emcee(params=result_all.params, is_weighted=True, float_behavior='chi2', **mdict)
                logger.debug('End EMCEE after %d iterations, redChi2 = %.3f',result_all.nfev,result_all.redchi) 
                # save fit emcee result to tabspec
                tabspec['ALL_FIT_EMCEE'] = model(result_all.params, wave_rest, family_lines, redshift, lsf)
            # save results
            logger.debug('Saving results to tablines and ztab')
            add_result_to_tables(result_all, tablines, ztab, redshift, sel_lines, lsf, init_minsnr, vac)
            resfit['lmfit_all'] = result_all         
            resfit['lines'] = tablines
            resfit['ztable'] = ztab
            if 'ALL_FIT_EMCEE' in tabspec.colnames:
                tabspec['LINE_FIT'] = tabspec['LINE_FIT'] + tabspec['ALL_FIT_EMCEE'] 
            else:
                tabspec['LINE_FIT'] = tabspec['LINE_FIT'] + tabspec['ALL_FIT_LSQ'] 
            tabspec['INIT_FIT'] = tabspec['INIT_FIT'] + tabspec['ALL_INIT_FIT']
            resfit['table_spec'] = tabspec 
            
        return resfit
    
    # we perform separate fitting for each families (except lyalpha already done)
    lines = lines[lines['LINE'] != 'LYALPHA']
    
    # fitting of families with non resonnant lines
    families = set(lines['FAMILY'])-set([3])
    logger.debug('Found %d non resonnant line families to fit', len(families))
    
    for id_family in families:
        family = family_names[id_family]
        logger.debug('Performing fitting of family %s', family)
        sel_lines = lines[lines['FAMILY']==id_family]
        logger.debug('LSQ Fitting of %d lines', len(sel_lines))
        # Set input parameters
        params = Parameters()
        set_gaussian_fitpars(family, params, sel_lines, line_ratios, 
                             redshift, lsf, init_vel, init_vdisp, init_windmax,
                             wave_rest, data_rest)
        family_lines = {family: {'fun':'gauss', 'lines':sel_lines['LINE']}}
        # Perform LSQ fit    
        minner = Minimizer(residuals, params, fcn_args=(wave_rest, data_rest, std_rest, family_lines, redshift, lsf))            
        logger.debug('Leastsq fitting with ftol: %.0e xtol: %.0e maxfev: %d',lsq_kws['ftol'],lsq_kws['xtol'],lsq_kws['maxfev'])
        result = minner.minimize(**lsq_kws)
        logger.debug('%s after %d iterations, redChi2 = %.3f',result.message,result.nfev,result.redchi)
        # save fit init and result to tabspec
        tabspec[f'{family.upper()}_INIT_FIT'] = model(params, wave_rest, family_lines, redshift, lsf)
        tabspec[f'{family.upper()}_FIT_LSQ'] = model(result.params, wave_rest, family_lines, redshift, lsf)            
        # Perform MCMC fit
        if emcee: 
            update_bounds(result, init_delta_vel, init_delta_vdisp, init_delta_gamma)
            mdict = set_nwakers(mcmc_kws, result)
            logger.debug('Error estimation using EMCEE with nsteps: %d nwalkers: %d burn: %d',mdict['steps'],mdict['nwalkers'],mdict['burn'])
            result = minner.emcee(params=result.params, is_weighted=True, float_behavior='chi2', **mdict)
            logger.debug('End EMCEE after %d iterations, redChi2 = %.3f',result.nfev,result.redchi) 
            # save fit emcee result to tabspec
            tabspec[f'{family.upper()}_FIT_EMCEE'] = model(result.params, wave_rest, family_lines, redshift, lsf)
        # save results
        logger.debug('Saving results to tablines and ztab')
        add_result_to_tables(result, tablines, ztab, redshift, sel_lines, lsf, init_minsnr, vac)
        resfit[f'lmfit_{family}'] = result         
        resfit['lines'] = tablines
        resfit['ztable'] = ztab
        if f'{family.upper()}_FIT_EMCEE' in tabspec.colnames:
            tabspec['LINE_FIT'] = tabspec['LINE_FIT'] + tabspec[f'{family.upper()}_FIT_EMCEE']
        else:
            tabspec['LINE_FIT'] = tabspec['LINE_FIT'] + tabspec[f'{family.upper()}_FIT_LSQ']
        tabspec['INIT_FIT'] = tabspec['INIT_FIT'] + tabspec[f'{family.upper()}_INIT_FIT']
        resfit['table_spec'] = tabspec 
        
    # fitting of families with resonnant lines (except lya, already fitted)
    lines = lines[lines['FAMILY']==3]
    dlines = reorganize_doublets(lines)    
    logger.debug('Found %d resonnant line families to fit', len(dlines)) 
    for clines in dlines:
        family = clines[0].lower()
        logger.debug('Performing fitting of family %s', family)
        ksel = lines['LINE']==clines[0]
        if len(clines) > 1:
            ksel = (lines['LINE']==clines[0]) | (lines['LINE']==clines[1])
        sel_lines = lines[ksel]           
        logger.debug('LSQ Fitting of %s', clines) 
        # Set input parameters
        params = Parameters()
        set_gaussian_fitpars(family, params, sel_lines, line_ratios, 
                             redshift, lsf, init_vel, init_vdisp, init_windmax,
                             wave_rest, data_rest)
        family_lines = {family: {'fun':'gauss', 'lines':sel_lines['LINE']}}
        # Perform LSQ fit    
        minner = Minimizer(residuals, params, fcn_args=(wave_rest, data_rest, std_rest, family_lines, redshift, lsf))            
        logger.debug('Leastsq fitting with ftol: %.0e xtol: %.0e maxfev: %d',lsq_kws['ftol'],lsq_kws['xtol'],lsq_kws['maxfev'])
        result = minner.minimize(**lsq_kws)
        logger.debug('%s after %d iterations, redChi2 = %.3f',result.message,result.nfev,result.redchi)
        # save fit init and result to tabspec
        tabspec[f'{family.upper()}_INIT_FIT'] = model(params, wave_rest, family_lines, redshift, lsf)
        tabspec[f'{family.upper()}_FIT_LSQ'] = model(result.params, wave_rest, family_lines, redshift, lsf)            
        # Perform MCMC fit
        if emcee: 
            update_bounds(result, init_delta_vel, init_delta_vdisp, init_delta_gamma)
            mdict = set_nwakers(mcmc_kws, result)
            logger.debug('Error estimation using EMCEE with nsteps: %d nwalkers: %d burn: %d',mdict['steps'],mdict['nwalkers'],mdict['burn'])
            result = minner.emcee(params=result.params, is_weighted=True, float_behavior='chi2', **mdict)
            logger.debug('End EMCEE after %d iterations, redChi2 = %.3f',result.nfev,result.redchi) 
            # save fit emcee result to tabspec
            tabspec[f'{family.upper()}_FIT_EMCEE'] = model(result.params, wave_rest, family_lines, redshift, lsf)
        # save results
        logger.debug('Saving results to tablines and ztab')
        add_result_to_tables(result, tablines, ztab, redshift, sel_lines, lsf, init_minsnr, vac)
        resfit[f'lmfit_{family}'] = result         
        resfit['lines'] = tablines
        resfit['ztable'] = ztab   
        if f'{family.upper()}_FIT_EMCEE' in tabspec.colnames:
            tabspec['LINE_FIT'] = tabspec['LINE_FIT'] + tabspec[f'{family.upper()}_FIT_EMCEE']
        else:
            tabspec['LINE_FIT'] = tabspec['LINE_FIT'] + tabspec[f'{family.upper()}_FIT_LSQ']
        tabspec['INIT_FIT'] = tabspec['INIT_FIT'] + tabspec[f'{family.upper()}_INIT_FIT']
        resfit['table_spec'] = tabspec 
          
    return resfit

def reorganize_doublets(lines):
    dlines = [[e['LINE']] for e in lines[lines['DOUBLET']==0]]
    doublets = lines[lines['DOUBLET']>0]
    ndoublets = set(doublets['DOUBLET'])
    for dlbda in ndoublets:
        slines = doublets[np.abs(doublets['DOUBLET']-dlbda) < 0.01]
        dlines.append([e['LINE'] for e in slines])
    return dlines
    

def add_result_to_tables(result, tablines, ztab, zinit, inputlines, lsf, snr_min, vac):
    """ add results to the table, if row exist it is updated"""
    par = result.params
    families = [key.split('_')[1] for key in par.keys() if key.split('_')[0]=='dv']
    for family in families: 
        dv = par[f"dv_{family}"].value
        dv_err = par[f"dv_{family}"].stderr
        vdisp = par[f"vdisp_{family}"].value 
        vdisp_err = par[f"vdisp_{family}"].stderr 
    
        lines = [key.split('_')[1] for key in par.keys() if (key.split('_')[0]==family) and (key.split('_')[3]=='l0')]
        flux_vals = []
        err_vals = []
        line_vals = []
        for line in lines:
            dname = inputlines[inputlines['LINE']==line]['DNAME'][0]
            keys = [key for key in par.keys() if key.split('_')[1]==line]
            fun = keys[0].split('_')[2]
            flux = par[f"{family}_{line}_{fun}_flux"].value
            flux_err = par[f"{family}_{line}_{fun}_flux"].stderr
            l0 = par[f"{family}_{line}_{fun}_l0"].value
            z = zinit+dv/C
            l1 = l0*(1+z)
            if not vac:
                l1 = vactoair(l1)
            lvals = {'LBDA_REST':l0, 'LBDA_OBS':l1, 'FLUX':flux, 
                     'DNAME':dname,  'VDISP':vdisp, 
                     'RCHI2':result.redchi, 'Z_INIT':zinit,  
                     }  
            if vdisp_err is not None:
                lvals['VDISP_ERR'] = vdisp_err 
            if flux_err is not None:
                lvals['FLUX_ERR'] = flux_err 
                lvals['SNR'] = flux/flux_err 
                flux_vals.append(flux)
                err_vals.append(flux_err)
                line_vals.append(line)
            if lsf:
                lvals['VDINST'] = complsf(l1, kms=True)             
            
            if fun == 'gauss':         
                sigma = get_sigma(vdisp, l1, z, lsf, restframe=False)
                fwhm = 2.355*sigma
                lvals.update({'FWHM_OBS':fwhm, 'LBDA_LEFT':l1-0.5*fwhm, 'LBDA_RIGHT':l1+0.5*fwhm, 
                         'PEAK_OBS':flux/(SQRT2PI*sigma), 'VEL':dv, 'Z':z}) 
                if dv_err is not None:
                    lvals['VEL_ERR'] = dv_err
                    lvals['Z_ERR'] = dv_err/C                
            elif fun == 'asymgauss':
                skew = par[f"{family}_{line}_{fun}_asym"].value
                lvals['SKEW'] = skew
                skew_err = par[f"{family}_{line}_{fun}_asym"].stderr 
                if skew_err is not None:
                    lvals['SKEW_ERR'] = skew_err
                swave_rest = np.linspace(l0-50,l0+50,1000)
                vmodel_rest = model_asymgauss(zinit, lsf, l0, flux, skew, vdisp, dv, swave_rest)
                kmax = np.argmax(vmodel_rest)    
                l1 = swave_rest[kmax]
                left_rest,right_rest = rest_fwhm_asymgauss(swave_rest, vmodel_rest)
                # these position is used for redshift and dv
                dv = C*(l1-l0)/l0
                lvals['VEL'] = dv
                if dv_err is not None:
                    lvals['VEL_ERR'] = dv_err
                    lvals['Z_ERR'] = dv_err/C                 
                z = zinit + dv/C
                lvals['Z'] = z  
                # compute the peak value and convert it to observed frame    
                lvals['PEAK_OBS'] = np.max(vmodel_rest)/(1+z)
                # save peak position in observed frame
                if vac:
                    lvals['LBDA_OBS'] = l1*(1+z)
                    lvals['LBDA_LEFT'] = left_rest*(1+z)
                    lvals['LBDA_RIGHT'] = right_rest*(1+z)                    
                else:
                    lvals['LBDA_OBS'] = vactoair(l1*(1+z))
                    lvals['LBDA_LEFT'] = vactoair(left_rest*(1+z))
                    lvals['LBDA_RIGHT'] = vactoair(right_rest*(1+z))
                lvals['FWHM_OBS'] = lvals['LBDA_RIGHT'] - lvals['LBDA_LEFT']                               
                
            upsert_ltable(tablines, lvals, family, line)

        zvals = {'VEL':dv, 'VDISP':vdisp, 'Z':zinit+dv/C,
                 'NFEV':result.nfev, 'RCHI2':result.redchi, 'Z_INIT':zinit,
                 } 
        if dv_err is not None:
            zvals['VEL_ERR'] = dv_err  
            zvals['Z_ERR'] = dv_err/C
        if vdisp_err is not None:
            zvals['VDISP_ERR'] = vdisp_err            
        if len(flux_vals) > 0:
            flux_vals = np.array(flux_vals)
            err_vals = np.array(err_vals)
            zvals['SNRSUM'] = np.sum(flux_vals)/np.sqrt(np.sum(err_vals**2))
            snr_vals = flux_vals/err_vals
            kmax = np.argmax(snr_vals)
            zvals['SNRMAX'] = snr_vals[kmax]
            zvals['LINE'] = line_vals[kmax]
            ksel = snr_vals > snr_min
            nl_clip = np.sum(ksel)
            if nl_clip > 0:
                zvals['SNRSUM_CLIPPED'] = np.sum(flux_vals[ksel])/np.sqrt(np.sum(err_vals[ksel]**2))
            zvals['NL'] = len(lines)
            zvals['NL_CLIPPED'] = nl_clip

        upsert_ztable(ztab, zvals, family)
        tablines.sort('LBDA_REST')
        
        
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
    
        
def add_gaussian_par(params, family_name, name, l0, z, lsf, wind_max, wave, data):
    params.add(f"{family_name}_{name}_gauss_l0", value=l0, vary=False)  
    ksel = np.abs(wave-l0) < wind_max
    vmax = data[ksel].max()
    sigma = get_sigma(0, l0, z, lsf, restframe=True)                  
    flux = SQRT2PI*sigma*vmax
    params.add(f"{family_name}_{name}_gauss_flux", value=flux, min=0)
    
def add_asymgauss_par(params, family_name, name, l0, z, lsf, wind_max, gamma, wave, data):
    params.add(f"{family_name}_{name}_asymgauss_l0", value=l0, vary=False)  
    ksel = np.abs(wave-l0) < wind_max
    vmax = data[ksel].max()
    sigma = get_sigma(0, l0, z, lsf, restframe=True)                  
    flux = SQRT2PI*sigma*vmax
    params.add(f"{family_name}_{name}_asymgauss_flux", value=flux, min=0)
    params.add(f"{family_name}_{name}_asymgauss_asym", value=gamma[1], 
               min=gamma[0], max=gamma[2])
    
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
                         init_dv, windmax, wave, data):
    logger = logging.getLogger(__name__)
    # add velocity and velocity dispersion fit parameters
    params.add(f'dv_{family_name}', value=init_vel[1], min=init_vel[0], max=init_vel[2])
    params.add(f'vdisp_{family_name}', value=init_dv[1], min=init_dv[0], max=init_dv[2]) 
    # we use gaussian parameters
    nc = 0
    for line in lines:      
        name = line['LINE']
        l0 = line['LBDA_REST']
        add_gaussian_par(params, family_name, name, l0, z, lsf, windmax, wave, data)
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
    params.add(f'dv_{family_name}', value=init_vel[1], min=init_vel[0], max=init_vel[2])
    params.add(f'vdisp_{family_name}', value=init_dv[1], min=init_dv[0], max=init_dv[2]) 
    # we use asymetric gaussian parameters
    nc = 0
    for line in lines:      
        name = line['LINE']
        l0 = line['LBDA_REST']
        add_asymgauss_par(params, family_name, name, l0, z, lsf, windmax, init_gamma, wave, data)
    logger.debug('added %d asymetric gaussian to the fit', len(lines))

          
    
def add_line_ratio(family, params, line_ratios, dlines):
    for line1,line2,ratio_min,ratio_max in line_ratios:
        if (line1 in dlines['LINE']) and (line2 in dlines['LINE']):
            params.add("%s_to_%s_factor" % (line1, line2), min=ratio_min,
                       max=ratio_max, value=0.5*(ratio_min+ratio_max))
            params['%s_%s_gauss_flux' % (family, line2)].expr = (
                "%s_%s_gauss_flux * %s_to_%s_factor" % (family, line1, line1, line2)
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
                flux = params[f"{name}_{line}_gauss_flux"].value
                l0 = params[f"{name}_{line}_gauss_l0"].value
                model += model_gauss(z, lsf, l0, flux, vdisp, dv, wave)
            elif ldict['fun']=='asymgauss':
                flux = params[f"{name}_{line}_asymgauss_flux"].value
                l0 = params[f"{name}_{line}_asymgauss_l0"].value
                beta = params[f"{name}_{line}_asymgauss_asym"].value  
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

def set_nwakers(d, result):
    """ set nwalkers for mcmc"""
    nwalkers = d.get('nwalkers', 0)
    if nwalkers == 0:               
        nwalkers = int(np.ceil(3*result.nvarys/2)*2) # nearest even number to 3*nb of variables
        mcdict = d.copy()
        mcdict['nwalkers'] = nwalkers
        return mcdict
    else:
        return d
    
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

def update_bounds(result, delta_vel, delta_vdisp, delta_gamma):
    """ update bounds with offset wrt result of current fit
    """
    logger = logging.getLogger(__name__)
    if all(v is None for v in [delta_vel,delta_vdisp,delta_gamma]):
        return
    
    par = result.params
    families = [key.split('_')[1] for key in par.keys() if key.split('_')[0]=='dv']
    for family in families: 
        if delta_vel is not None:
            dv = par[f"dv_{family}"].value
            par[f"dv_{family}"].min = dv - delta_vel
            par[f"dv_{family}"].max = dv + delta_vel
        if delta_vdisp is not None:
            vdisp = par[f"vdisp_{family}"].value
            par[f"vdisp_{family}"].min = vdisp - delta_vdisp
            par[f"vdisp_{family}"].max = vdisp + delta_vdisp 
        if delta_gamma is not None:
            fun = [key.split('_')[2] for key in par.keys() if (key.split('_')[0]==family) and (key.split('_')[3]=='l0')][0]
            if fun == 'asymgauss':
                lines = [key.split('_')[1] for key in par.keys() if (key.split('_')[0]==family) and (key.split('_')[3]=='l0')]
                for line in lines:
                    gamma = par[f"{family}_{line}_{fun}_asym"].value
                    par[f"{family}_{line}_{fun}_asym"].min = gamma - delta_gamma
                    par[f"{family}_{line}_{fun}_asym"].max = gamma + delta_gamma
 
    logger.debug('Update bounds relative to LSQ fit. delta vel %s vdisp %s gamma %s',delta_vel,delta_vdisp,delta_gamma)    
    
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
