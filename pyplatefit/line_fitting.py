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
from scipy.special import erf
from scipy.signal import argrelmin
from logging import getLogger

from mpdaf.sdetect.linelist import get_emlines
from mpdaf.obj.spectrum import vactoair, airtovac

C = constants.c.to(u.km / u.s).value
SQRT2PI = np.sqrt(2*np.pi)

# Parameters used in the fitting
VEL_MIN, VEL_INIT, VEL_MAX = -500, 0, 500  # Velocity
VD_MIN, VD_INIT, VD_MAX = 50, 80, 300  # Velocity dispersion
VD_MAX_LYA = 700  # Maximum velocity dispersion for Lyman α
GAMMA_MIN, GAMMA_INIT, GAMMA_MAX = -5, 0, 5  # γ parameter for Lyman α
WINDOW_MAX = 30 # search radius in A for peak around starting wavelength


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
    

    def fit(self, line_spec, z, major_lines=False, lines=None, emcee=False, use_line_ratios=True,
            vel_uniq_offset=False):
        """
        perform line fit on a mpdaf spectrum
        
        """
        if use_line_ratios:
            # we use a default for OII and CIII
            line_ratios = DOUBLET_RATIOS
        else:
            line_ratios = None
        fit_kws = dict(maxfev=self.maxfev, xtol=self.xtol, ftol=self.ftol)
        fit_lws = dict(vel=self.vel, vdisp=self.vdisp, vdisp_lya_max=self.vdisp_lya_max, gamma=self.gamma)
        return fit_mpdaf_spectrum(line_spec, z, major_lines=major_lines, lines=lines, emcee=emcee, line_ratios=line_ratios,
                                  vel_uniq_offset=vel_uniq_offset, fit_kws=fit_kws, fit_lws=fit_lws)
    
    def info(self, res):
        #if res.get('spec', None) is not None:
            #if hasattr(res['spec'], 'filename'):
                #self.logger.info(f"Spectrum: {res['spec'].filename}")
                
        report_fit(res)
            
        #self.logger.info(f"Line Fit Status: {res['ier']} {res['mesg']} Niter: {res['nfev']}")
        #self.logger.info(f"Line Fit Chi2: {res['redchi']:.2f} Bic: {res['bic']:.2f}")
        #self.logger.info(f"Line Fit Z: {res['z']:.5f} Err: {res['z_err']:.5f} dZ: {res['dz']:.5f}")
        #self.logger.info(f"Line Fit dV: {res['v']:.2f} Err: {res['v_err']:.2f} km/s Bounds [{res['v_min']:.0f} : {res['v_max']:.0f}] Init: {res['v_init']:.0f} km/s")
        #self.logger.info(f"Line Fit Vdisp: {res['vdisp']:.2f} Err: {res['vdisp_err']:.2f} km/s Bounds [{res['vdisp_min']:.0f} : {res['vdisp_max']:.0f}] Init: {res['vdisp_init']:.0f} km/s")
        
        #if res.get('v_lyalpha', None) is not None:
            #self.logger.info(f"Line Fit Z Lyalpha: {res['zlya']:.5f} Err: {res['zlya_err']:.5f} dZ: {res['dzlya']:.5f}")
            #self.logger.info(f"Line Fit dV Lyalpha: {res['v_lyalpha']:.2f} Err: {res['v_lyalpha_err']:.2f} km/s Bounds [{res['v_lyalpha_min']:.0f} : {res['v_lyalpha_max']:.0f}] Init: {res['v_lyalpha_init']:.0f} km/s")
            #self.logger.info(f"Line Fit Vdisp Lyalpha: {res['vdisp_lyalpha']:.2f} Err: {res['vdisp_lyalpha_err']:.2f} km/s Bounds [{res['vdisp_lyalpha_min']:.0f} : {res['vdisp_lyalpha_max']:.0f}] Init: {res['vdisp_lyalpha_init']:.0f} km/s")
            #self.logger.info(f"Line Fit Skewness Lyalpha: {res['skewlya']:.2f} Err: {res['skewlya_err']:.2f} Bounds [{res['skewlya_min']:.0f} : {res['skewlya_max']:.0f}] Init: {res['skewlya_init']:.0f}")
        

        

            

        


    


def fit_spectrum_lines(wave, data, std, redshift, *, unit_wave=None,
                       unit_data=None, vac=False, lines=None, line_ratios=None,
                       major_lines=False, emcee=False,
                       vel_uniq_offset=False, 
                       fit_kws=None, fit_lws=None):
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
    - DNAME: The display name for the line (set to None for close doublets)
    - VEL: The velocity offset in km/s with respect to the initial redshift (rest frame)
    - VEL_ERR: The error in velocity offset in km/s 
    - Z: The fitted redshift in vacuum of the line (note for lyman-alpha the line peak is used)
    - Z_ERR: The error in fitted redshift of the line.
    - VDISP: The fitted velocity dispersion in km/s (rest frame)
    - VDISP_ERR: The error in fitted velocity dispersion
    - FLUX: Flux in the line. The unit depends on the units of the spectrum.
    - FLUX_ERR: The fitting uncertainty on the flux value.
    - SKEW: The skewness of the asymetric line (for Lyman-alpha line only).
    - SKEW_ERR: The uncertainty on the skewness (for Lyman-alpha line only).
    - LBDA_OBS: The fitted position the line in the observed frame
    - PEAK_OBS: The fitted peak of the line in the observed frame
    - FWHM_OBS: The full width at half maximum of the line in the observed frame 
    
  
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
    vel_uniq_offset: boolean, optional
        if True, use same velocity offset for all lines (not recommended)
        if False, allow different velocity offsets between balmer, forbidden and resonnant lines
        default: false 
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
    
    
    if vel_uniq_offset:
        families = [[[1,2,3],'all']]
    else:
        # fit different velocity and velocity dispersion for balmer and forbidden family 
        families = [[[1],'balmer'],[[2],'forbidden']]

        
    for family_ids,family_name in families:
        ksel = lines['FAMILY']==family_ids[0]
        if len(family_ids)>1:
            for family_id in family_ids[1:]:
                ksel = ksel | (lines['FAMILY']==family_id)
        sel_lines = lines[ksel]
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
            add_gaussian_par(params, family_name, name, l0, wave_rest, data_rest)
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
            logger.debug('%d %s lines to fit', len(sel_lines), family_name)
            doublets = sel_lines[sel_lines['DOUBLET']>0]
            singlets = sel_lines[sel_lines['DOUBLET']==0]
            if len(singlets) > 0:
                for line in singlets:
                    if line['LINE'] == 'LYALPHA':
                        # we fit an asymmetric line
                        fname = line['LINE'].lower()
                        family_lines[fname] = dict(lines=[line['LINE']], fun='asymgauss')
                        params.add(f'dv_{fname}', value=VEL_INIT, min=VEL_MIN, max=VEL_MAX)
                        params.add(f'vdisp_{fname}', value=VD_INIT, min=VD_MIN, max=VD_MAX_LYA) 
                        name = line['LINE']
                        l0 = line['LBDA_REST']
                        add_asymgauss_par(params, fname, name, l0, wave_rest, data_rest)
                    else:
                        # we fit a gaussian
                        fname = line['LINE'].lower()
                        family_lines[fname] = dict(lines=[line['LINE']], fun='gauss')
                        params.add(f'dv_{fname}', value=VEL_INIT, min=VEL_MIN, max=VEL_MAX)
                        params.add(f'vdisp_{fname}', value=VD_INIT, min=VD_MIN, max=VD_MAX)
                        name = line['LINE']
                        l0 = line['LBDA_REST']
                        add_gaussian_par(params, fname, name, l0, wave_rest, data_rest)
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
                        add_gaussian_par(params, fname, name, l0, wave_rest, data_rest)
                    if line_ratios is not None:
                        # add line ratios bounds
                        add_line_ratio(params, line_ratios, dlines)
 
                    
        
    minner = Minimizer(residuals, params, fcn_args=(wave_rest, data_rest, std_rest, family_lines))
    
    logger.debug('Leastsq fitting')
    result = minner.minimize()
    logger.debug('%s after %d iterations, redChi2 = %.3f',result.message,result.nfev,result.redchi)
    
    if emcee:
        logger.debug('Error estimation using EMCEE')
        result = minner.emcee(params=result.params, is_weighted=True)
        logger.debug('End EMCEE after %d iterations, redChi2 = %.3f',result.nfev,result.redchi)
    
    # save input data, initial and best fit (in rest frame)
    result.wave = wave_rest
    result.data = data_rest
    result.std = std_rest
    result.init = model(params, wave_rest, family_lines)
    result.fit = model(result.params, wave_rest, family_lines)
    
    # fill the lines table with the fit results
    lines.remove_columns(['LBDA_LOW','LBDA_UP','TYPE','DOUBLET','FAMILY','LBDA_EXP'])
    for colname in ['VEL','VEL_ERR','Z','Z_ERR','VDISP','VDISP_ERR',
                    'FLUX','FLUX_ERR','SNR','SKEW','SKEW_ERR','LBDA_OBS','PEAK_OBS','FWHM_OBS']:
        lines.add_column(MaskedColumn(name=colname, dtype=np.float, length=len(lines), mask=True))
    for colname in ['VEL','VEL_ERR','VDISP','VDISP_ERR','SNR',
                    'FLUX','FLUX_ERR','SKEW','SKEW_ERR','LBDA_OBS','PEAK_OBS','FWHM_OBS']:
        lines[colname].format = '.2f'
    lines['Z'].format = '.5f'
    lines['Z_ERR'].format = '.2e'
   
    
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
            row['VEL'] = dv
            row['VEL_ERR'] = dv_err
            row['Z'] = redshift + dv/C
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
            sigma = vdisp*row['LBDA_OBS']/C
            peak = flux/(SQRT2PI*sigma)
            row['PEAK_OBS'] = peak
            row['FWHM_OBS'] = 2.355*sigma 
            if fun == 'asymgauss':
                skew = par[f"{name}_{fun}_asym"].value 
                row['SKEW'] = skew 
                row['SKEW_ERR'] = par[f"{name}_{fun}_asym"].stderr if par[f"{name}_{fun}_asym"].stderr is not None else np.nan
                # compute peak location and peak value in rest frame  
                l0 = row['LBDA_REST']    
                sigma = vdisp*l0/C
                l1 = mode_skewedgaussian(l0, sigma, skew)
                # these position is used for redshift and dv
                dv = C*(l1-l0)/l0
                row['VEL'] = dv
                row['Z'] = redshift + dv/C
                # compute the peak value and convert it to observed frame
                peak = flux/(SQRT2PI*sigma)
                ksel = np.abs(wave_rest-l0)<50
                vmodel = asymgauss(peak, l0, sigma, skew, wave_rest[ksel])
                row['PEAK_OBS'] = np.max(vmodel)/(1+row['Z'])
                # compute FWHM
                fwhm = measure_fwhm(wave_rest[ksel], vmodel, l1)   
                row['FWHM_OBS'] = fwhm*(1+row['Z'])
                # save peak position in observed frame
                row['LBDA_OBS'] = vactoair(l1*(1+row['Z']))
                
  
    # save line table
    result.linetable = lines
    
    return result

def add_gaussian_par(params, family_name, name, l0, wave, data):
    params.add(f"{name}_gauss_l0", value=l0, vary=False)  
    ksel = np.abs(wave-l0) < WINDOW_MAX
    vmax = data[ksel].max()
    sigma = VD_INIT*l0/C
    flux = SQRT2PI*sigma*vmax
    params.add(f"{name}_gauss_flux", value=flux, min=0)
    
def add_asymgauss_par(params, family_name, name, l0, wave, data):
    params.add(f"{name}_asymgauss_l0", value=l0, vary=False)  
    ksel = np.abs(wave-l0) < WINDOW_MAX
    vmax = data[ksel].max()
    sigma = VD_INIT*l0/C
    flux = SQRT2PI*sigma*vmax
    params.add(f"{name}_asymgauss_flux", value=flux, min=0)
    params.add(f"{name}_asymgauss_asym", value=GAMMA_INIT, min=GAMMA_MIN, max=GAMMA_MAX)
    
def add_line_ratio(params, line_ratios, dlines):
    for line1,line2,ratio_min,ratio_max in line_ratios:
        if (line1 in dlines['LINE']) and (line2 in dlines['LINE']):
            params.add("%s_to_%s_factor" % (line1, line2), min=ratio_min,
                       max=ratio_max, value=0.5*(ratio_min+ratio_max))
            params['%s_gauss_flux' % line2].expr = (
                "%s_gauss_flux * %s_to_%s_factor" % (line1, line1, line2)
            )    

def model(params, wave, lines):
    model = 0
    for name,ldict in lines.items():
        vdisp = params[f"vdisp_{name}"]
        dv = params[f"dv_{name}"]
        for line in ldict['lines']:
            if ldict['fun']=='gauss':
                flux = params[f"{line}_gauss_flux"]
                l0 = params[f"{line}_gauss_l0"]*(1+dv/C)
                sigma = vdisp*l0/C
                peak = flux/(SQRT2PI*sigma)
                model += gauss(peak, l0, sigma, wave)
            elif ldict['fun']=='asymgauss':
                flux = params[f"{line}_asymgauss_flux"]
                l0 = params[f"{line}_asymgauss_l0"]*(1+dv/C)
                beta = params[f"{line}_asymgauss_asym"].value
                sigma = vdisp*l0/C
                peak = flux/(SQRT2PI*sigma)
                model += asymgauss(peak, l0, sigma, beta, wave)            
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

def asymgauss(peak, l0, sigma, gamma, wave):
    dl = wave - l0
    g = peak*np.exp(-dl**2/(2*sigma**2))
    f = 1 + erf(gamma*dl/(1.4142135623730951*sigma))
    h = f*g
    return h


def fit_mpdaf_spectrum(spectrum, redshift, major_lines=False, lines=None, emcee=False, line_ratios=None,
                       vel_uniq_offset=False, fit_kws={}, fit_lws={}):
    """Function use when calling fit_lines from mpdaf spectrum object.


    Parameters
    ----------
    spectrum : mpdaf.obj.Spectrum
    redshift : float
    major_lines : boolean
       if true, use only major lines as defined in MPDAF line list
    lines: list
       list of MPDAF lines to use in the fit
       default None
    emcee: boolean
       if True perform a second fit using EMCEE to derive improved errors (note cpu intensive)
       default False
    line_ratios: list
       list of constrained line ratios (see fit-spectrum_lines)
       default None

    Returns
    -------
    See fit_spectrum_lines.
    return in addition the fitted spectrum in the observed frame

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
            

    res = fit_spectrum_lines(wave=wave, data=data, std=std, redshift=redshift,
                             unit_wave=u.angstrom, unit_data=unit_data, line_ratios=line_ratios,
                             lines=lines, major_lines=major_lines, emcee=emcee,
                             vel_uniq_offset=vel_uniq_offset,
                             fit_kws=fit_kws, fit_lws=fit_lws)
    
    # add fitted spectra on the observed plane
    spfit = spectrum.clone()
    # convert wave to observed frame and air
    wave = res.wave*(1 + redshift)
    wave = vactoair(wave)
    spfit.data = np.interp(spectrum.wave.coord(), wave, res.fit)
    spfit.data = spfit.data / (1 + redshift)
    
    res.spec_fit = spfit
            
    return res

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
