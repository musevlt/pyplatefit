###############################################################################
#
# Copyright (C) 2016, Michele Cappellari
# E-mail: michele.cappellari_at_physics.ox.ac.uk
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
###############################################################################

# This file contains the 'miles' class with functions to contruct a
# library of MILES templates and interpret and display the output
# of pPXF when using those templates as input.

from __future__ import print_function

from os import path
import glob

import numpy as np
from scipy import ndimage
from astropy.io import fits

import ppxf_util as util
from cap_readcol import readcol


###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 27 November 2016

def age_metal(filename):
    """
    Extract the age and metallicity from the name of a file of
    the MILES library of Single Stellar Population models as
    downloaded from http://miles.iac.es/ as of 2016

    :param filename: string possibly including full path
        (e.g. 'miles_library/Mun1.30Zm0.40T03.9811.fits')
    :return: age (Gyr), [M/H]

    """
    s = path.basename(filename)
    age = float(s[s.find("T")+1:s.find(".fits")])
    metal = s[s.find("Z")+1:s.find("T")]
    if "m" in metal:
        metal = -float(metal[1:])
    elif "p" in metal:
        metal = float(metal[1:])
    else:
        raise ValueError("This is not a standard MILES filename")

    return age, metal

###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Adapted from my procedure setup_spectral_library() in
#       ppxf_example_population_sdss(), to make it a stand-alone procedure.
#     - Read the characteristics of the spectra directly from the file names
#       without the need for the user to edit the procedure when changing the
#       set of models. Michele Cappellari, Oxford, 28 November 2016
#   V1.0.1: Check for files existence. MC, Oxford, 31 March 2017

class miles(object):

    def __init__(self, pathname, velscale, FWHM_gal,
                        FWHM_tem=2.51, normalize=False):
        """
        Produces an array of logarithmically-binned templates by reading
        the spectra from the Single Stellar Population (SSP) library by
        Vazdekis et al. (2010, MNRAS, 404, 1639) http://miles.iac.es/.
        The code checks that the model specctra form a rectangular grid
        in age and metallicity and properly sorts them in both parameters.
        The code also returns the age and metallicity of each template
        by reading these parameters directly from the file names.
        The templates are broadened by a Gaussian with dispersion
        sigma_diff = np.sqrt(sigma_gal**2 - sigma_tem**2).

        Thie script relies on the files naming convention adopted by
        the MILES library, where SSP spectra have the form below

            Mun1.30Zm0.40T03.9811.fits

        This code can be easily adapted by the users to deal with other stellar
        libraries, different IMFs or different abundances.

        :param pathname: path with wildcards returning the list files to use
            (e.g. 'miles_models/Mun1.30*.fits'). The files must form a Cartesian grid
            in age and metallicity and the procedure returns an error if they are not.
        :param velscale: desired velocity scale for the output templates library in km/s
            (e.g. 60). This is generally the same or an integer fraction of the velscale
            of the galaxy spectrum.
        :param FWHM_gal: vector or scalar of the FWHM of the instrumental resolution of
            the galaxy spectrum in Angstrom.
        :param normalize: set to True to normalize each template to mean=1.
            This is useful to compute light-weighted stellar population quantities.
        :return: The following variables are stored as attributes of the miles class:
            .templates: array has dimensions templates[npixels, n_ages, n_metals];
            .log_lam_temp: natural np.log() wavelength of every pixel npixels;
            .age_grid: (Gyr) has dimensions age_grid[n_ages, n_metals];
            .metal_grid: [M/H] has dimensions metal_grid[n_ages, n_metals].
            .n_ages: number of different ages
            .n_metal: number of different metallicities
        """
        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname

        all = [age_metal(f) for f in files]
        all_ages, all_metals = np.array(all).T
        ages, metals = np.unique(all_ages), np.unique(all_metals)
        n_ages, n_metal = len(ages), len(metals)

        assert set(all) == set([(a, b) for a in ages for b in metals]), \
            'Ages and Metals do not form a Cartesian grid'

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the SDSS galaxy spectrum, to determine
        # the size needed for the array which will contain the template spectra.
        hdu = fits.open(files[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam_range_temp = h2['CRVAL1'] + np.array([0, h2['CDELT1']*(h2['NAXIS1']-1)])
        sspNew, log_lam_temp = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[:2]

        templates = np.empty((sspNew.size, n_ages, n_metal))
        age_grid = np.empty((n_ages, n_metal))
        metal_grid = np.empty((n_ages, n_metal))

        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the galaxy and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
        sigma = FWHM_dif/2.355/h2['CDELT1']   # Sigma difference in pixels

        # Here we make sure the spectra are sorted in both [M/H] and Age
        # along the two axes of the rectangular grid of templates.
        for j, age in enumerate(ages):
            for k, metal in enumerate(metals):
                p = all.index((age, metal))
                hdu = fits.open(files[p])
                ssp = hdu[0].data
                if np.isscalar(FWHM_gal):
                    ssp = ndimage.gaussian_filter1d(ssp, sigma)
                else:
                    ssp = util.gaussian_filter1d(ssp, sigma)  # convolution with variable sigma
                sspNew = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[0]
                if normalize:
                    sspNew /= np.mean(sspNew)
                templates[:, j, k] = sspNew
                age_grid[j, k] = age
                metal_grid[j, k] = metal

        self.templates = templates/np.median(templates)  # Normalize by a scalar
        self.log_lam_temp = log_lam_temp
        self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.n_ages = n_ages
        self.n_metal = n_metal


###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 1 December 2016
#   V1.0.1: Use path.realpath() to deal with symbolic links.
#       Thanks to Sam Vaughan (Oxford) for reporting problems.
#       MC, Garching, 11/JAN/2016

    def mass_to_light(self, weights, band="r", quiet=False):
        """
        Computes the M/L in a chosen band, given the weights produced
        in output by pPXF. A Salpeter IMF is assumed (slope=1.3).

        This procedure uses the photometric predictions
        from Vazdekis+12 and Ricciardelli+12
        http://adsabs.harvard.edu/abs/2012MNRAS.424..157V
        http://adsabs.harvard.edu/abs/2012MNRAS.424..172R
        they were downloaded in December 2016 below and are included in pPXF with permission
        http://www.iac.es/proyecto/miles/pages/photometric-predictions/based-on-miuscat-seds.php

        :param weights: pPXF output with dimensions weights[miles.n_ages, miles.n_metal]
        :param band: possible choices are "U", "B", "V", "R", "I", "J", "H", "K" for
            the Vega photometric system and "u", "g", "r", "i" for the SDSS AB system.
        :param quiet: set to True to suppress the printed output.
        :return: mass_to_light in the given band
        """
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        vega_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
        sdss_bands = ["u", "g", "r", "i"]
        vega_sun_mag = [5.600, 5.441, 4.820, 4.459, 4.148, 3.711, 3.392, 3.334]
        sdss_sun_mag = [6.55, 5.12, 4.68, 4.57]  # values provided by Elena Ricciardelli

        file_dir = path.dirname(path.realpath(__file__))  # path of this procedure

        if band in vega_bands:
            k = vega_bands.index(band)
            sun_mag = vega_sun_mag[k]
            file2 = file_dir + "/miles_models/Vazdekis2012_ssp_phot_Padova00_UN_v10.0.txt"
        elif band in sdss_bands:
            k = sdss_bands.index(band)
            sun_mag = sdss_sun_mag[k]
            file2 = file_dir + "/miles_models/Vazdekis2012_ssp_sdss_miuscat_UN1.30_v9.txt"
        else:
            raise ValueError("Unsupported photometric band")

        file1 = file_dir + "/miles_models/Vazdekis2012_ssp_mass_Padova00_UN_baseFe_v10.0.txt"
        slope1, MH1, Age1, m_no_gas = readcol(file1, usecols=[1, 2, 3, 5])

        slope2, MH2, Age2, mag = readcol(file2, usecols=[1, 2, 3, 4 + k])

        # The following loop is a brute force but very safe and general
        # way of matching the photometric quantities to the SSP spectra.
        # It makes no assumption on the sorting and dimensions of the files
        mass_no_gas_grid = np.empty_like(weights)
        lum_grid = np.empty_like(weights)
        for j in range(self.n_ages):
            for k in range(self.n_metal):
                p1 = (np.abs(self.age_grid[j, k] - Age1) < 0.001) & \
                     (np.abs(self.metal_grid[j, k] - MH1) < 0.01) & \
                     (np.abs(1.30 - slope1) < 0.01)
                mass_no_gas_grid[j, k] = m_no_gas[p1]

                p2 = (np.abs(self.age_grid[j, k] - Age2) < 0.001) & \
                     (np.abs(self.metal_grid[j, k] - MH2) < 0.01) & \
                     (np.abs(1.30 - slope2) < 0.01)
                lum_grid[j, k] = 10**(-0.4*(mag[p2] - sun_mag))

        # This is eq.(2) in Cappellari+13
        # http://adsabs.harvard.edu/abs/2013MNRAS.432.1862C
        mlpop = np.sum(weights*mass_no_gas_grid)/np.sum(weights*lum_grid)

        if not quiet:
            print('M/L_' + band + ': %.4g' % mlpop)

        return mlpop


###############################################################################

    def plot(self, weights, nodots=False, colorbar=True, **kwargs):

        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        xgrid = np.log10(self.age_grid) + 9
        ygrid = self.metal_grid
        util.plot_weights_2d(xgrid, ygrid, weights,
                             nodots=nodots, colorbar=colorbar, **kwargs)


##############################################################################

    def mean_age_metal(self, weights, quiet=False):

        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        log_age_grid = np.log10(self.age_grid) + 9
        metal_grid = self.metal_grid

        # These are eq.(1) and (2) in McDermid+15
        # http://adsabs.harvard.edu/abs/2015MNRAS.448.3484M
        mean_log_age = np.sum(weights*log_age_grid)/np.sum(weights)
        mean_metal = np.sum(weights*metal_grid)/np.sum(weights)

        if not quiet:
            print('Weighted <logAge> [yr]: %.3g' % mean_log_age)
            print('Weighted <[M/H]>: %.3g' % mean_metal)

        return mean_log_age, mean_metal


##############################################################################
