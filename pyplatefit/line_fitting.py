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

from astropy import constants
from astropy import units as u
from astropy.table import Table
from astropy.table import MaskedColumn
from lmfit.parameter import Parameters
from lmfit import Minimizer, report_fit
import numpy as np
from scipy.signal import argrelmin
from logging import getLogger

from mpdaf.sdetect.linelist import get_emlines
from mpdaf.obj.spectrum import vactoair, airtovac

C = constants.c.to(u.km / u.s).value

# Parameters used in the fitting
VEL_MIN, VEL_INIT, VEL_MAX = -500, 0, 500  # Velocity
VD_MIN, VD_INIT, VD_MAX = 50, 80, 300  # Velocity dispersion
VD_MAX_LYA = 700  # Maximum velocity dispersion for Lyman α
GAMMA_MIN, GAMMA_INIT, GAMMA_MAX = -5, 0, 5  # γ parameter for Lyman α


DOUBLET_RATIOS = [
    ("CIII1907", "CIII1909", 0.6, 1.2),
    ("OII3727", "OII3729", 1.0, 2.0),
]


class NoLineError(ValueError):
    """Error raised when there is no line to fit in the spectrum."""

    pass


class Linefit:
    """
    This class implement Emission Line def
    """
    def __init__(self, vel=(-500,0,500), vdisp=(50,80,300), vdisp_lya_max=700, gamma_lya=(-5,0,5), xtol=1.e-4, ftol=1.e-6, maxfev=1000):
        self.logger = getLogger(__name__)
        self.maxfev = maxfev # nb max of iterations (leastsq)
        self.xtol = xtol # relative error in the solution (leastq)
        self.ftol = ftol # relative error in the sum of square (leastsq)
        self.vel = vel
        self.vdisp = vdisp
        self.vdisp_lya_max = vdisp_lya_max
        self.gamma = gamma_lya
       
        return
    

    def fit(self, line_spec, z, major_lines=False, lines=None, emcee=False):
        """
        perform line fit on a mpdaf spectrum
        
        """
        fit_kws = dict(maxfev=self.maxfev, xtol=self.xtol, ftol=self.ftol)
        fit_lws = dict(vel=self.vel, vdisp=self.vdisp, vdisp_lya_max=self.vdisp_lya_max, gamma=self.gamma)
        return fit_mpdaf_spectrum(line_spec, z, major_lines=major_lines, lines=lines, emcee=emcee, 
                                  fit_kws=fit_kws, fit_lws=fit_lws)
    
    def info(self, res):
        if res.get('spec', None) is not None:
            if hasattr(res['spec'], 'filename'):
                self.logger.info(f"Spectrum: {res['spec'].filename}")
            
        self.logger.info(f"Line Fit Status: {res['ier']} {res['mesg']} Niter: {res['nfev']}")
        self.logger.info(f"Line Fit Chi2: {res['redchi']:.2f} Bic: {res['bic']:.2f}")
        self.logger.info(f"Line Fit Z: {res['z']:.5f} Err: {res['z_err']:.5f} dZ: {res['dz']:.5f}")
        self.logger.info(f"Line Fit dV: {res['v']:.2f} Err: {res['v_err']:.2f} km/s Bounds [{res['v_min']:.0f} : {res['v_max']:.0f}] Init: {res['v_init']:.0f} km/s")
        self.logger.info(f"Line Fit Vdisp: {res['vdisp']:.2f} Err: {res['vdisp_err']:.2f} km/s Bounds [{res['vdisp_min']:.0f} : {res['vdisp_max']:.0f}] Init: {res['vdisp_init']:.0f} km/s")
        
        if res.get('v_lyalpha', None) is not None:
            self.logger.info(f"Line Fit Z Lyalpha: {res['zlya']:.5f} Err: {res['zlya_err']:.5f} dZ: {res['dzlya']:.5f}")
            self.logger.info(f"Line Fit dV Lyalpha: {res['v_lyalpha']:.2f} Err: {res['v_lyalpha_err']:.2f} km/s Bounds [{res['v_lyalpha_min']:.0f} : {res['v_lyalpha_max']:.0f}] Init: {res['v_lyalpha_init']:.0f} km/s")
            self.logger.info(f"Line Fit Vdisp Lyalpha: {res['vdisp_lyalpha']:.2f} Err: {res['vdisp_lyalpha_err']:.2f} km/s Bounds [{res['vdisp_lyalpha_min']:.0f} : {res['vdisp_lyalpha_max']:.0f}] Init: {res['vdisp_lyalpha_init']:.0f} km/s")
            self.logger.info(f"Line Fit Skewness Lyalpha: {res['skewlya']:.2f} Err: {res['skewlya_err']:.2f} Bounds [{res['skewlya_min']:.0f} : {res['skewlya_max']:.0f}] Init: {res['skewlya_init']:.0f}")
        

        

            
    def eqw(self):
        """
        compute equivalent widths
        """
        
    def snr(self):
        """
        compute  SNR
        
        """

        

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

def fit_spectrum_lines(wave, data, std, redshift, *, unit_wave=None,
                       unit_data=None, vac=False, lines=None, line_ratios=None,
                       major_lines=False, fit_kws=None, fit_lws=None, emcee=False):
    """Fit lines in a spectrum using lmfit.

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
    - LINE: the name of the line
    - LBDA_REST: The the rest-frame position of the line in vacuum
    - VEL: The velocity offset in km/s with respect to the initial redshift
    - VEL_ERR: The error in velocity offset in km/s 
    - Z: The fitted redshift in vacuum of the line (note for lyman-alpha the line peak is used)
    - Z_ERR: The error in fitted redshift of the line.
    - LBDA_EXP: The input position of the line, i.e. the rest-frame position
      redshifted using the input redshift and if needed converted to air.
    - LBDA: The fitted position the line in the observed frame
    - LBDA_ERR: The uncertainty on the position of the LINE. 
    - FWHM: The full width at half maximum of the line in the observed frame and converted to km/s.
    - FWHM_ERR: The uncertainty on the FWHM in the observed frame in km/s. 
    - SKEW: The skewness of the asymetric line (for Lyman-alpha line only).
    - SKEW_ERR: The uncertainty on the skewness (for Lyman-alpha line only).
    - FLUX: Flux in the line. The unit depends on the units of the spectrum.
      The function try to find the corresponding unit.
    - FLUX_ERR: The fitting uncertainty on the flux value.


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
          lines that are expected in the spectrum will be fitted. Note that the
          names of the resonant lines in the LINE column is important an must
          match the names in RESONANT_LINES.
    line_ratios: list of (str, str, float, float) tuples or string
        List on line ratio constraints (line1 name, line2 name, line2/line1
        minimum, line2/line1 maximum.
    major_lines : boolean, optional
        If true, the fit is restricted to the major lines as defined in mpdaf line table (used only when lines is None, )
        default: False
    emcee : boolean, optional
        if true, errors and best fit is estimated with EMCEE starting from the leastsq solution
        default: False
    fit_kws : dictionary with leasq parameters (see scipy.optimize.leastsq)
    fit_kws : dictionary with some default and bounds parameters

    Returns
    -------
    result_dict : OrderedDict
        Dictionary containing several parameters from the fitting.

    Raises
    ------
    NoLineError: when none of the fitted line can be on the spectrum at the
        given redshift.
    """
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
        lines = get_emlines(z=redshift, vac=True, sel=sel, margin=5,
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

    # The fitting is done with lmfit. The model is a sum of Gaussian (or
    # skewed Gaussian), one per line.
    params = Parameters()  # All the parameters of the fit
    family_lines = {} # dictionary of family lines
    
    # fit one unique velocity and velocity dispersion for balmer and forbidden family
    for family_id,family_name in [[1,'balmer'],[2,'forbidden']]:
        sel_lines = lines[lines['FAMILY']==family_id]
        if len(sel_lines) == 0:
            logger.debug('No %s lines to fit', family_name)
            continue
        else:
            logger.debug('%d %s lines to fit', len(sel_lines), family_name)
        family_lines[family_name] = dict(lines=sel_lines['LINE'].tolist(), fun='gauss')
        params.add(f'dv_{family_name}', value=0, min=-500, max=500)
        params.add(f'vdisp_{family_name}', value=50, min=30, max=100)
        for line in sel_lines:
            name = line['LINE']
            params.add(f"{name}_gauss_l0", value=line['LBDA_REST'], vary=False)  
            ksel = np.abs(wave_rest-line['LBDA_REST']) < 30
            vmax = data_rest[ksel].max() 
            params.add(f"{name}_gauss_peak", value=vmax, min=0)
        if line_ratios is not None:
            # add line ratios bounds
            dlines = sel_lines[sel_lines['DOUBLET']>0]
            if len(dlines) == 0:
                continue
            for line1,line2,ratio_min,ratio_max in line_ratios:
                if (line1 in dlines['LINE']) and (line2 in dlines['LINE']):
                    params.add("%s_to_%s_factor" % (line1, line2), min=ratio_min,
                               max=ratio_max, value=0.5*(ratio_min+ratio_max))
                    params['%s_gauss_peak' % line2].expr = (
                        "%s_gauss_peak * %s_to_%s_factor" % (line1, line1, line2)
                    ) 
                    logger.debug('Add doublet constrain for %s %s min %.2f max %.2f',line1,line2,ratio_min,ratio_max)
        
    # fit a different velocity and velocity dispersion for each resonnant lines(or doublet)
    family_id = 3
    sel_lines = lines[lines['FAMILY']==family_id]
    if len(sel_lines) == 0:
        logger.debug('No %s lines to fit', family_name)
    else:   
        logger.debug('%d %s lines to fit', len(sel_lines), family_name)
        doublets = sel_lines[sel_lines['DOUBLET']>0]
        singlets = sel_lines[sel_lines['DOUBLET']==0]
        if len(singlets) > 0:
            for line in singlets:
                if line['LINE'] == 'LYALPHA':
                    # we fit an asymmetric line
                    fname = line['LINE'].lower()
                    family_lines[fname] = dict(lines=[line['LINE']], fun='asymgauss')
                    params.add(f'dv_{fname}', value=0, min=-500, max=500)
                    params.add(f'vdisp_{fname}', value=100, min=30, max=300) 
                    name = line['LINE']
                    params.add(f"{name}_asymgauss_l0", value=line['LBDA_REST'], vary=False)  
                    ksel = np.abs(wave_rest-line['LBDA_REST']) < 30
                    vmax = data_rest[ksel].max() 
                    params.add(f"{name}_asymgauss_peak", value=vmax, min=0)
                    params.add(f"{name}_asymgauss_asym", value=0) 
                else:
                    # we fit a gaussian
                    fname = line['LINE'].lower()
                    family_lines[fname] = dict(lines=[line['LINE']], fun='gauss')
                    params.add(f'dv_{fname}', value=0, min=-500, max=500)
                    params.add(f'vdisp_{fname}', value=50, min=30, max=100) 
                    name = line['LINE']
                    params.add(f"{name}_gauss_l0", value=line['LBDA_REST'], vary=False)  
                    ksel = np.abs(wave_rest-line['LBDA_REST']) < 30
                    vmax = data_rest[ksel].max() 
                    params.add(f"{name}_gauss_peak", value=vmax, min=0)
        if len(doublets) > 0:
            ndoublets = np.unique(doublets['DOUBLET'])
            for dlbda in ndoublets:
                dlines = doublets[np.abs(doublets['DOUBLET']-dlbda) < 0.01]
                fname = str(dlines['LINE'][0]).lower()
                family_lines[fname] = dict(lines=dlines['LINE'], fun='gauss')
                params.add(f'dv_{fname}', value=0, min=-500, max=500)
                params.add(f'vdisp_{fname}', value=50, min=30, max=100)                 
                for line in dlines:
                    name = line['LINE']
                    params.add(f"{name}_gauss_l0", value=line['LBDA_REST'], vary=False)  
                    ksel = np.abs(wave_rest-line['LBDA_REST']) < 30
                    vmax = data_rest[ksel].max() 
                    params.add(f"{name}_gauss_peak", value=vmax, min=0)
                if line_ratios is not None:
                    # add line ratios bounds
                    for line1,line2,ratio_min,ratio_max in line_ratios:
                        if (line1 in dlines['LINE']) and (line2 in dlines['LINE']):
                            params.add("%s_to_%s_factor" % (line1, line2), min=ratio_min,
                                       max=ratio_max, value=0.5*(ratio_min+ratio_max))
                            params['%s_gauss_peak' % line2].expr = (
                                "%s_gauss_peak * %s_to_%s_factor" % (line1, line1, line2)
                            ) 
                            logger.debug('Add doublet constrain for %s %s min %.2f max %.2f',line1,line2,ratio_min,ratio_max)                
                    
        
    minner = Minimizer(residuals, params, fcn_args=(wave_rest, data_rest, std_rest, family_lines))
    
    logger.debug('Leastsq fitting')
    result = minner.minimize()
    
    if emcee:
        logger.debug('Error estimation using EMCEE')
        result = minner.emcee(params=result.params)
    
    # save input data, initial and best fit (in rest frame)
    result.wave = wave_rest
    result.data = data_rest
    result.std = std_rest
    result.init = model(params, wave_rest, family_lines)
    result.fit = model(result.params, wave_rest, family_lines)
    
    # fill the lines table with the fit results
    lines.remove_columns(['LBDA_LOW','LBDA_UP','TYPE','DOUBLET','FAMILY'])
    for colname in ['VEL','VEL_ERR','Z','Z_ERR','LBDA','LBDA_ERR','FLUX','FLUX_ERR',
                    'PEAK','PEAK_ERR','FWHM','FWHM_ERR','FWHMOBS','FWHMOBS_ERR','SKEW','SKEW_ERR']:
        lines.add_column(MaskedColumn(name=colname, dtype=np.float, length=len(lines), mask=True))
    
    par = result.params
    zf = 1+redshift
    for fname,fdict in family_lines.items():
        dv = par[f"dv_{fname}"].value * zf
        dv_err = par[f"dv_{fname}"].stderr * zf
        vdisp = par[f"vdisp_{fname}"].value * zf
        vdisp_err = par[f"vdisp_{fname}"].stderr * zf  
        fwhm = 2.355*vdisp
        fwhm_err = 2.355*vdisp_err
        fun = fdict['fun']
        for row in lines:
            name = row['LINE']
            if name not in fdict['lines']:
                continue
            peak = par[f"{name}_{fun}_peak"].value / zf
            if par[f"{name}_{fun}_peak"].expr is None:
                peak_err = par[f"{name}_{fun}_peak"].stderr / zf
            else: #to be compute with factor
                peak_err = 0
                
            row['VEL'] = dv
            row['VEL_ERR'] = dv_err
            row['Z'] = redshift + dv/C
            row['Z_ERR'] = dv_err/C
            row['LBDA'] = row['LBDA_REST']*(1+row['Z'])
            if not vac:
                row['LBDA'] = vactoair(row['LBDA'])
            row['LBDA_ERR'] = row['LBDA_REST']*row['Z_ERR']
            row['PEAK'] = peak
            row['PEAK_ERR'] = peak_err
            row['FWHM'] = fwhm
            row['FWHM_ERR'] = fwhm_err
            row['FWHMOBS'] = fwhm*row['LBDA']/C
            row['FWHMOBS_ERR'] = fwhm_err*row['LBDA']/C 
            if fun == 'gauss':
                flux = 2*np.pi*row['FWHMOBS']*peak/2.355
                flux_err = (2*np.pi/2.355) * (row['FWHMOBS_ERR']*peak + row['FWHMOBS']*peak_err)
                row['FLUX'] = flux
                row['FLUX_ERR'] = flux_err
            if fun == 'asymgauss':
                row['SKEW'] = par[f"{name}_{fun}_asym"].value 
                row['SKEW_ERR'] = par[f"{name}_{fun}_asym"].stderr
                flux = peak*get_asym_flux(par[f"{name}_{fun}_asym"], vdisp*row['LBDA']/C)
                fwhm = get_asym_fwhm(par[f"{name}_{fun}_asym"].value, vdisp*row['LBDA']/C)
                row['FLUX'] = flux
                row['FWHMOBS'] = fwhm*(wave[1]-wave[0])
                row['FWHM'] = row['FWHMOBS']*C/row['LBDA']
        
    
    # save line table
    result.linetable = lines
    
    return result

def get_asym_flux(a,d):
    from scipy.integrate import quad
    f = lambda x: np.exp(-x**2/(2*(a*x+d)**2))
    flux = quad(f, -np.inf, np.inf)
    return flux[0]

def get_asym_fwhm(a,d):
    from scipy.optimize import brentq
    f = lambda x: np.exp(-x**2/(2*(a*x+d)**2))-0.5
    x = brentq(f, 0, 5*d)
    return 2*x

def model(params, wave, lines):
    model = 0
    for name,ldict in lines.items():
        vdisp = params[f"vdisp_{name}"]
        dv = params[f"dv_{name}"]
        for line in ldict['lines']:
            if ldict['fun']=='gauss':
                peak = params[f"{line}_gauss_peak"]
                l0 = params[f"{line}_gauss_l0"]*(1+dv/C)
                sigma = vdisp*l0/C
                model += gauss(peak.value, l0, sigma, wave)
            elif ldict['fun']=='asymgauss':
                peak = params[f"{line}_asymgauss_peak"]
                l0 = params[f"{line}_asymgauss_l0"]*(1+dv/C)
                asym = params[f"{line}_asymgauss_asym"].value
                sigma = vdisp*l0/C
                model += asymgauss(peak.value, l0, asym, sigma, wave)            
            else:
                logger.error('Unknown function %s', fun)
                raise ValueError
    return model
    
def residuals(params, wave, data, std, lines):
    vmodel = model(params, wave, lines)
    res = (vmodel - data)/std 
    return res
    
def gauss(peak, l0, sigma, wave):
    g = peak*np.exp(-(wave-l0)**2/(2*sigma**2))
    return g

def asymgauss(peak, l0, a, d, wave):
    sigma = a*(wave-l0) + d
    g = peak*np.exp(-(wave-l0)**2/(2*sigma**2))
    return g


def fit_spectrum_lines_old(wave, data, std, redshift, *, unit_wave=None,
                       unit_data=None, vac=False, lines=None, line_ratios=None,
                       snr_width=None, force_positive_fluxes=False,
                       trim_spectrum=True, return_lmfit_info=False,
                       fit_kws=None, fit_lws=None):
 
    logger = logging.getLogger(__name__)

    wave = np.array(wave)
    data = np.array(data)
    # lmfit minimizes (data-model)*weight is the least square sense. So
    # the weight must be the inverse of the uncertainty.
    if std is not None:
        weights = 1 / np.array(std)
    else:
        weights = None

    # We may want to compute the spectrum noise on part of the spectrum removed
    # while trimming it. So we keep a copy of the wavelength.
    orig_wave = wave.copy()

    # Unit of the computed flux.
    if unit_wave is not None and unit_data is not None:
        # The flux is the integral of the data in the line profile.
        unit_flux = unit_data * unit_wave
    else:
        unit_flux = None

    # The fitting is done with lmfit. The model is a sum of Gaussian (or
    # skewed Gaussian), one per line.
    params = Parameters()  # All the parameters of the fit
    model_list = []  # List of per line model.

    # Fitting only some lines from mpdaf library.
    if type(lines) is list:
        lines_to_fit = lines
        lines = None
    else:
        lines_to_fit = None

    if lines is None:
        logger.debug("Getting lines from get_emlines...")
        lines = get_emlines(z=redshift, vac=vac,
                            lbrange=[wave.min(), wave.max()],
                            ltype="em", table=True)
        lines.rename_column("LBDA_OBS", "LBDA_EXP")
        if lines_to_fit is not None:
            lines = lines[np.in1d(lines['LINE'], lines_to_fit)]
            if len(lines) < len(lines_to_fit):
                logger.debug(
                    "Some lines are not on the spectrum coverage: %s.",
                    ", ".join(set(lines_to_fit) - set(lines['LINE'])))
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
        raise NoLineError("There is no know line on the spectrum "
                          "coverage.")
    
    # get defaut parameters from fit_lws or common default values
    if (fit_lws is None):
        pvel_min,pvel_init,pvel_max = (VEL_MIN,VEL_INIT,VEL_MAX) 
        pvd_min,pvd_init,pvd_max = (VD_MIN,VD_INIT,VD_MAX) 
        pvd_max_lya = VD_MAX_LYA
        pgamma_min,pgamma_init,pgamma_max = (GAMMA_MIN,GAMMA_INIT,GAMMA_MAX)
    else:
        pvel_min,pvel_init,pvel_max = fit_lws['vel']
        pvd_min,pvd_init,pvd_max = fit_lws['vdisp']
        pvd_max_lya = fit_lws['vdisp_lya_max'] 
        pgamma_min,pgamma_init,pgamma_max = fit_lws['gamma'] 

    # Spectrum trimming
    # The window we keep around each line depend on the minimal and maximal
    # velocity (responsible for shifting the line), and on the maximal velocity
    # dispersion (responsible for the spreading of the line). We add a 3σ
    # margin.
    if trim_spectrum:
        mask = np.full_like(wave, False, dtype=bool)  # Points to keep
        for row in lines:
            line_wave = row["LBDA_EXP"]
            if row['LINE'] == "LYALPHA":
                    vd_max = pvd_max_lya
            else:
                    vd_max = pvd_max
            wave_min = line_wave * (1 + pvd_min / C)
            wave_min -= 3 * wave_min * vd_max / C
            wave_max = line_wave * (1 + pvel_max / C)
            wave_max += 3 * wave_max * vd_max / C
            mask[(wave >= wave_min) & (wave <= wave_max)] = True
            logger.debug("Keeping only waves in [%s, %s] for line %s.",
                         wave_min, wave_max, row['LINE'])
        wave, data, weights = wave[mask], data[mask], weights[mask]
        logger.debug("%.1f %% of the spectrum is used for fitting.",
                     100 * np.sum(mask) / len(mask))
    # mask all points that have a weight == 0
    mask = weights <= 0
    if np.sum(mask) > 0:
        logger.debug('Masked %d points with weights <= 0', np.sum(mask))
        wave, data, weights = wave[~mask], data[~mask], weights[~mask]

    # If there are non resonant lines, we add the velocity parameters that
    # will be used in the Gaussian models.
    if set(lines['LINE']) - set(RESONANT_LINES):
        logger.debug("Adding velocity to parameters...")
        # Velocity of the galaxy in km/s. Accounts as uncertainty on the
        # redshift.
        params.add("v", pvel_init, min=pvel_min, max=pvel_max)
        # Velocity dispersion of the galaxy in km/s.
        params.add("vdisp", pvd_init, min=pvd_min, max=pvd_max)

    # If there are resonant lines, we add one set of velocity parameters
    # per element (some resonant lines may be doublets).
    for elem in set([RESONANT_LINES[l] for l in lines['LINE'] if l in
                     RESONANT_LINES]):
        logger.debug("Adding velocities for %s...", elem)
        params.add("v_%s" % elem, pvel_init, min=pvel_min, max=pvel_max)
        params.add("vdisp_%s" % elem, pvd_init, min=pvd_min, max=pvd_max)

    # For the Lyman α line, we allow of a greater velocity dispersion and
    # compute a Lyman α redshift.
    if "LYALPHA" in lines['LINE']:
        logger.debug("Adding Lyman α velocity dispersion...")
        params['vdisp'].max = pvd_max_lya

    # Per line model creation
    for row in lines:
        line_name = row["LINE"]
        line_wave = row["LBDA_EXP"]

        # Depending on the type of line, we don't use the same velocity
        # parameter in the model.
        if line_name in RESONANT_LINES:
            res_prefix = RESONANT_LINES[line_name]
            velocity_param = "v_%s" % res_prefix
            velocity_disp_param = "vdisp_%s" % res_prefix
        else:
            velocity_param = "v"
            velocity_disp_param = "vdisp"

        # We model each line with a Gaussian except for the Lyman α line
        # which is modelled with a skewed Gaussian.
        if line_name != "LYALPHA":
            line_model = GaussianModel(prefix=line_name + "_")
        else:
            line_model = SkewedGaussianModel(prefix="LYALPHA_")

        # The model is appended to the list of models and its parameters
        # are added to the main parameter dictionary.
        model_list.append(line_model)
        params += line_model.make_params()

        # The amplitude of the line is the area of the Gaussian (it
        # measures the flux). As starting value, we take the sum of the
        # spectrum in a window around the expected wavelength; the window is
        # bigger for Lyman α because this line can go quite far from where it
        # is expected.
        radius = 20 if line_name == "LYALPHA" else 5
        mask = (wave > line_wave - radius) & (wave < line_wave + radius)
        line_start_value = np.sum(data[mask])
        if force_positive_fluxes and line_start_value < 0:
            line_start_value = 0
        params["%s_amplitude" % line_name].value = line_start_value 
        if force_positive_fluxes:
            params["%s_amplitude" % line_name].min = 0

        # Limit the γ parameter of the Lyman α line
        if line_name == "LYALPHA":
            params["LYALPHA_gamma"].value = pgamma_init
            params["LYALPHA_gamma"].min = pgamma_min
            params["LYALPHA_gamma"].max = pgamma_max

        # The center of the Gaussian is parameterized with the initial line
        # wavelength modified by the velocity:
        # μ = λ0 * (1 + v/c)
        params["%s_center" % line_name].expr = (
            "%s * (1 + %s / %s)" % (line_wave, velocity_param, C))

        # The sigma of the Gaussian is parameterized with the center of the
        # line and the velocity dispersion.
        # σ = μ * vd/c
        params["%s_sigma" % line_name].expr = (
            "%s_center * %s / %s" % (line_name, velocity_disp_param, C))

        logger.debug("Line %s at %s Angstroms with %s as starting value added",
                     line_name, line_wave, line_start_value)

    # Line ratio constraints
    if line_ratios is not None:
        for line1, line2, ratio_min, ratio_max in line_ratios:
            if line1 in lines['LINE'] and line2 in lines['LINE']:
                logger.debug("Adding %s / %s flux ration constraint.", line2,
                             line1)

                # If the ratio of the initial value for the line amplitudes is
                # within the bounds, we use this value as starting point. If
                # not, we use the mean of the bounds and update the initial
                # value of the second line accordingly.
                try:
                    ratio_init = (params['%s_amplitude' % line2].value /
                                  params['%s_amplitude' % line1].value)
                except ZeroDivisionError:
                    ratio_init = np.mean([ratio_min, ratio_max])
                    # Initial value of line 1 is set accorting to the initial
                    # ratio.
                    params['%s_amplitude' % line1].value = (
                        params['%s_amplitude' % line2].value / ratio_init
                    )
                if not (ratio_min <= ratio_init <= ratio_max):
                    ratio_init = np.mean([ratio_min, ratio_max])
                    # Initial value of line 2 is set accorting to the initial
                    # ratio.
                    params['%s_amplitude' % line2].value = (
                        ratio_init * params['%s_amplitude' % line1].value)

                params.add("%s_to_%s_factor" % (line1, line2), min=ratio_min,
                           max=ratio_max, value=ratio_init)
                params['%s_amplitude' % line2].expr = (
                    "%s_amplitude * %s_to_%s_factor" % (line1, line1, line2)
                )

    # Global model. Using sum() does not work here.
    model = model_list.pop()
    while model_list:
        model += model_list.pop()
        

    logger.debug("Fitting the lines...")
    lmfit_results = model.fit(data, params, x=wave, weights=weights, fit_kws=fit_kws)

    # Result parameters
    result_dict = OrderedDict()
    result_dict["ier"] = lmfit_results.ier
    result_dict["mesg"] = lmfit_results.message
    result_dict["nfev"] = lmfit_results.nfev
    result_dict["chisqr"] = lmfit_results.chisqr
    result_dict["redchi"] = lmfit_results.redchi
    result_dict["aic"] = lmfit_results.aic
    result_dict["bic"] = lmfit_results.bic
    for param in lmfit_results.params:
        if "v" in param or "factor" in param:
            result_dict[param] = lmfit_results.params[param].value
            try:
                result_dict["%s_err" % param] = \
                    float(lmfit_results.params[param].stderr)
            except TypeError:
                result_dict["%s_err" % param] = np.nan
            try:
                result_dict["%s_min" % param] = \
                    float(lmfit_results.params[param].min)
            except TypeError:
                result_dict["%s_min" % param] = -np.nan 
            try:
                result_dict["%s_max" % param] = \
                    float(lmfit_results.params[param].max)
            except TypeError:
                result_dict["%s_max" % param] = np.nan 
            result_dict["%s_init" % param] = lmfit_results.init_params[param].value
            
    if 'vdisp' in lmfit_results.params:
        result_dict['vdisp'] = lmfit_results.params['vdisp'].value
        result_dict['vdisp_err'] = lmfit_results.params['vdisp'].stderr if lmfit_results.params['vdisp'].stderr is not None else np.nan
        result_dict['vdisp_min'] = lmfit_results.params['vdisp'].min
        result_dict['vdisp_max'] = lmfit_results.params['vdisp'].max
        result_dict['vdisp_init'] = lmfit_results.init_params['vdisp'].value
    # New redshift taking into account the velocity
    # (1 + z_new) = (1 + z_ini)(1 + v/c)
    # Note: the redshift variable contains the initial redshift
    if "v" in lmfit_results.params:
        result_dict["z"] = (
            (1 + redshift) *
            (1 + lmfit_results.params['v'].value / C) - 1
        )
        try:
            result_dict["z_err"] = (
                (1 + redshift) *
                lmfit_results.params['v'].stderr / C
            )
        except TypeError:
            result_dict["z_err"] = np.nan
        result_dict["dz"] = result_dict['z'] - redshift
    if "v_lyalpha" in lmfit_results.params:
        result_dict["zlya"] = (
            (1 + redshift) *
            (1 + lmfit_results.params['v_lyalpha'].value / C) - 1
        )
        try:
            result_dict["zlya_err"] = (
                (1 + redshift) *
                lmfit_results.params['v_lyalpha'].stderr / C
            )
        except TypeError:
            result_dict["zlya_err"] = np.nan
        result_dict["dzlya"] = result_dict['zlya'] - redshift
    if "LYALPHA_gamma" in lmfit_results.params:
        result_dict["skewlya"] = (
            lmfit_results.params['LYALPHA_gamma'].value
        )
        result_dict["skewlya_err"] = (
            lmfit_results.params['LYALPHA_gamma'].stderr
        )
        result_dict["skewlya_min"] = lmfit_results.params['LYALPHA_gamma'].min
        result_dict["skewlya_max"] = lmfit_results.params['LYALPHA_gamma'].max
        result_dict["skewlya_init"] = lmfit_results.init_params['LYALPHA_gamma'].value
    
    # Line table
    l_name, l_lambda_exp, l_lambda, l_lambda_err, l_fwhm, l_fwhm_err, \
        l_value, l_std = [], [], [], [], [], [], [], []
    for row in lines:
        line_name = row['LINE']
        l_name.append(line_name)
        l_lambda_exp.append(row['LBDA_EXP'])
        if line_name != "LYALPHA":
            center = lmfit_results.params['%s_center' % line_name].value
            # Some times the error can't be calculated
            try:
                center_err = float(
                    lmfit_results.params['%s_center' % line_name].stderr)
            except TypeError:
                center_err = np.nan
            fwhm = C * 2 * np.sqrt(2 * np.log(2)) * (
                lmfit_results.params['%s_sigma' % line_name].value / center)
            try:
                fwhm_err = C * 2 * np.sqrt(2 * np.log(2)) * (
                    float(lmfit_results.params['%s_sigma' % line_name].stderr) /
                    center)
            except TypeError:
                fwhm_err = np.nan
        else:
            # Approximation of Lyman α maximum position.
            center = mode_skewedgaussian(
                lmfit_results.params['LYALPHA_center'].value,
                lmfit_results.params['LYALPHA_sigma'].value,
                lmfit_results.params['LYALPHA_gamma'].value
            )
            center_err = np.nan
            # Measure of the FWHM on the best fit model
            fwhm = C * (
                measure_fwhm(wave, lmfit_results.best_fit, center) /
                center)
            fwhm_err = np.nan
        l_lambda.append(center)
        l_lambda_err.append(center_err)
        l_fwhm.append(fwhm)
        l_fwhm_err.append(fwhm_err)
        l_value.append(
            lmfit_results.params['%s_amplitude' % line_name].value)
        # Some times the error can't be calculated
        try:
            l_std.append(float(
                lmfit_results.params['%s_amplitude' % line_name].stderr
            ))
        except TypeError:
            l_std.append(np.nan)

    line_table = Table(
        [l_name, l_lambda_exp, l_lambda, l_lambda_err, l_fwhm, l_fwhm_err,
         l_value, l_std],
        names=["LINE", "LBDA_EXP", "LBDA", "LBDA_ERR", "FWHM", "FWHM_err",
               "FLUX", "FLUX_ERR"],
    )

    if snr_width is not None:
        if std is None:
            raise ValueError("Can't compute a SNR without the spectrum "
                             "standard deviation.")

        logger.debug("Computing the SNR...")
        snr_col = []
        for row in line_table:
            # FWHM in wavelength unit
            fwhm_w = row['LBDA'] * row['FWHM'] / C
            logger.debug("Computing SNR of %s using a box of %s angstrom.",
                         row['LINE'], snr_width * fwhm_w)
            wave_min = row['LBDA'] - snr_width * fwhm_w / 2
            wave_max = row['LBDA'] + snr_width * fwhm_w / 2
            mask = (orig_wave >= wave_min) & (orig_wave <= wave_max)
            variance = np.trapz(y=std[mask]**2, x=orig_wave[mask])
            snr_col.append(np.abs(row['FLUX']) / np.sqrt(variance))
        line_table["SNR"] = snr_col

    line_table.sort("LBDA")
    line_table["LBDA_EXP"].unit = unit_wave
    line_table["LBDA"].unit = unit_wave
    line_table["LBDA_ERR"].unit = unit_wave
    line_table["FWHM"].unit = u.km / u.s
    line_table["FWHM_err"].unit = u.km / u.s
    line_table["FLUX"].unit = unit_flux
    line_table["FLUX_ERR"].unit = unit_flux
    
    result_dict['table'] = line_table
    
    
    if return_lmfit_info:
        # Add a wave attribute to lmfit_results for an easy access to the
        # wavelengths associated to the data in the fitting (useful when trimming
        # the spectrum).
        lmfit_results.wave = lmfit_results.userkws['x']
        result_dict['lmfit'] = lmfit_results
    

    return result_dict


def fit_mpdaf_spectrum(spectrum, redshift, major_lines=False, lines=None, emcee=False, line_ratios=None,
                       fit_kws={}, fit_lws={}):
    """Function use when calling fit_lines from mpdaf spectrum object.


    Parameters
    ----------
    spectrum : mpdaf.obj.Spectrum
    redshift : float
    return_lmfit_info : boolean
       if true, the bestfit spectrum is added in the result dictionary
    **kwargs : various
        Keyword arguments passed to fit_spectrum_lines.

    Returns
    -------
    See fit_spectrum_lines.

    """

    wave = spectrum.wave.coord(unit=u.angstrom).copy()
    data = spectrum.data

    if spectrum.var is not None:
        std = np.sqrt(spectrum.var)
    else:
        std = None

    # FIXME: ODHIN may produce spectra with the variance to 0 in some
    # points. Use infinite for these points so that they are not taken into
    # account.
    if std is not None:
        bad_points = std == 0
        std[bad_points] = np.inf

    try:
        unit_data = u.Unit(spectrum.data_header.get("BUNIT", None))
    except ValueError:
        unit_data = None
    
    if line_ratios is None:
        # we use a default for OII and CIII
        line_ratios = DOUBLET_RATIOS
        

    res = fit_spectrum_lines(wave=wave, data=data, std=std, redshift=redshift,
                             unit_wave=u.angstrom, unit_data=unit_data, line_ratios=line_ratios,
                             lines=lines, major_lines=major_lines, emcee=emcee,
                             fit_kws=fit_kws, fit_lws=fit_lws)
    
    

        
    return res
