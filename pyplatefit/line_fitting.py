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
from lmfit.models import GaussianModel, SkewedGaussianModel
from lmfit.parameter import Parameters
import numpy as np
from scipy.signal import argrelmin
from logging import getLogger

from mpdaf.sdetect.linelist import get_emlines
from mpdaf.obj.spectrum import vactoair

C = constants.c.to(u.km / u.s).value

# Parameters used in the fitting
VEL_MIN, VEL_INIT, VEL_MAX = -500, 0, 500  # Velocity
VD_MIN, VD_INIT, VD_MAX = 50, 80, 300  # Velocity dispersion
VD_MAX_LYA = 700  # Maximum velocity dispersion for Lyman α
GAMMA_MIN, GAMMA_INIT, GAMMA_MAX = -5, 0, 5  # γ parameter for Lyman α

# When fitting the lines, it is assumed that the resonant ones have the own
# velocity and velocity dispersion, but shared in the case of doublets. This
# dictionary associate the name of the line to the suffixed used in the
# parameters.
RESONANT_LINES = {
    "LYALPHA": "lyalpha",
    "MGII2796": "mgii",
    "MGII2803": "mgii",
}


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
    

    def fit(self, line_spec, z, return_lmfit_info=True, **kwargs):
        """
        perform line fit on a mpdaf spectrum
        
        """
        fit_kws = dict(maxfev=self.maxfev, xtol=self.xtol, ftol=self.ftol)
        fit_lws = dict(vel=self.vel, vdisp=self.vdisp, vdisp_lya_max=self.vdisp_lya_max, gamma=self.gamma)
        return fit_mpdaf_spectrum(line_spec, z, return_lmfit_info=return_lmfit_info, 
                                  fit_kws=fit_kws, fit_lws=fit_lws, **kwargs)
    
    def info(self, res):
        if res.get('spec', None) is not None:
            if hasattr(res['spec'], 'filename'):
                self.logger.info(f"Spectrum: {res['spec'].filename}")
            
        self.logger.info(f"Line Fit Status: {res['ier']} {res['mesg']} Niter: {res['nfev']}")
        self.logger.info(f"Line Fit Chi2: {res['redchi']:.2f} Bic: {res['bic']:.2f}")
        self.logger.info(f"Line Fit Z: {res['z']:.5f} Err: {res['z_err']:.5f} dZ: {res['dz']:.5f}")
        self.logger.info(f"Line Fit dV: {res['v']:.2f} Err: {res['v_err']:.2f} km/s Bounds [{res['v_min']:.0f} : {res['v_max']:.0f}] Init: {res['v_init']:.0f} km/s")
        self.logger.info(f"Line Fit Vdisp: {res['vdisp']:.2f} Err: {res['vdisp_err']:.2f} km/s Bounds [{res['vdisp_min']:.0f} : {res['vdisp_max']:.0f}] Init: {res['vdisp_init']:.0f} km/s")
        

        

            
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
                       snr_width=None, force_positive_fluxes=False,
                       trim_spectrum=True, return_lmfit_info=False,
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

    These values were computed on the DR1 catalog from MUSE-UDF ans are
    consistent with Pradhan et al. (2006) and Stark et al. (2015) given the
    kind of galaxies observed in MUSE-UDF.

    If the units of the wavelength and data axis are provided, the function
    will try to determine the unit of the computed fluxes.

    The second output of the function is a table of the lines found in the
    spectrum. The columns are:
    - LINE: the name of the line
    - LBDA_EXP: The expected position of the line, i.e. the rest-frame position
      redshifted and if needed converted to air.
    - LBDA: The position the line is fitted at.
    - LBDA_ERR: The uncertainty on the position of the LINE. Note that for
      Lyman α we don't provide this information.
    - FWHM: The full width at half maximum of the line converted to km/s.
    - FLUX: Flux in the line. The unit depends on the units of the spectrum.
      The function try to find the corresponding unit.
    - FLUX_ERR: The fitting uncertainty on the flux value.
    - SNR: The signal to noise ratio computed by measuring the flux uncertainty
      on the spectrum in a snr_width wide window.

    Note on errors: The FLUX_ERR column contains the error from the fitting
    procedure, which is different from the noise in the spectrum. It's
    nevertheless possible to ask for the computation of a signal to noise ratio
    (SNR) which is computed from the standard deviation associated to the
    spectrum.

    FIXME: The result dictionary `velocity_lyalpha` and
    `velocity_dispersion_lyalpha` parameters are the wrong one because they are
    based in the non-skewed Gaussian.

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
          column with the rest-frame, vaccuum wavelength of the lines. This
          line
          table will be redshifted and converted to air wavelengths. Only the
          lines that are expected in the spectrum will be fitted. Note that the
          names of the resonant lines in the LINE column is important an must
          match the names in RESONANT_LINES.
    line_ratios: list of (str, str, float, float) tuples or string
        List on line ratio constraints (line1 name, line2 name, line2/line1
        minimum, line2/line1 maximum.
    snr_width: float, optional
        Width, in FWHM factor, around the line position in which the noise is
        measured to compute the SNR of the line. If not provided, the SNR is
        not added to the line table.
    force_positive_fluxes : boolean, optional
        If true, the flux in the lines will be forced to be positive. Use
        carefully as this may lead to the impossibility to compute errors.
    trim_spectrum : boolean, optional
        If true, the fit is done keeping only the parts of the spectrum around
        the expected lines.
    return_lmfit_info: boolean, optional
        if true, the lmfit detailed return info is added in the return dictionary
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
    ValueError: if the computation of the FWHM is asked with the `snr_width`
        parameter but the spectrum has no variance.

    """
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
        pvd_max = fit_lws['vdisp_lya_max'] 
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
        params['vdisp_lya'].max = pvd_max_lya

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
        result_dict['vdisp_err'] = lmfit_results.params['vdisp'].stderr
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
    if "vlya" in lmfit_results.params:
        result_dict["zlya"] = (
            (1 + redshift) *
            (1 + lmfit_results.params['vlya'].value / C) - 1
        )
        try:
            result_dict["zlya_err"] = (
                (1 + redshift) *
                lmfit_results.params['vlya'].stderr / C
            )
        except TypeError:
            result_dict["zlya_err"] = np.nan
    if "LYALPHA_gamma" in lmfit_results.params:
        result_dict["skewlya"] = (
            lmfit_results.params['LYALPHA_gamma'].value
        )
        result_dict["skelya_err"] = (
            lmfit_results.params['LYALPHA_gamma'].stderr
        )
    
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


def fit_mpdaf_spectrum(spectrum, redshift, return_lmfit_info=False, **kwargs):
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

    res = fit_spectrum_lines(wave=wave, data=data, std=std, redshift=redshift,
                              unit_wave=u.angstrom, unit_data=unit_data,
                              return_lmfit_info=return_lmfit_info,
                              **kwargs)
    
    
    if return_lmfit_info:
        bestfit = spectrum.clone()
        bestfit.data = np.interp(spectrum.wave.coord(), res['lmfit'].wave, res['lmfit'].best_fit) 
        res['line_fit'] = bestfit
        res['line_spec'] = spectrum
        
    return res
