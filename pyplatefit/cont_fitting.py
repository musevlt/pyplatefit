import numpy as np
import os
import sys

from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits
from mpdaf.obj import airtovac, vactoair
from logging import getLogger

from .nnls_burst import fit_continuum1

CURDIR = os.path.dirname(os.path.abspath(__file__))

class Contfit:
    """
    This class is a poorman version of continuum fit Platefit
    """
    def __init__(self):
        self.logger = getLogger(__name__)
        # ---------------------------------------------- GENERAL SETTINGS --------------------------------------------------
        settings = {
            # directories
            'pipelinedir': CURDIR,
            'twoddir': '',
            'oneddir': '',
            'dustdir': '',
            'platefitdir': '',

            # line fit information
            'linepars': {},
            'indexpars': {},
            'obspars': {},
            'obstags': {},

            # burst model files
            'burst_model_file': 'BC03/bc_models_subset_cb08_milesx__bursts_extlam.fit',
    
            'available_z': [0.0001, 0.0004, 0.001,  0.004,  0.008,  0.017,  0.04,   0.07],
            'use_z': [0.0001, 0.0004, 0.001,  0.004,  0.008,  0.017,  0.04,   0.07 ],

            # burst models information to be recorded
            'burst_lib': [],
            'burst_wl': [],

            
            'gzval': [],
            'szval': [],
            'gztags': [],
            'sztags': [],
            'ssp_ages': [],
            'ssp_norms': [],
            'model_dispersion': [],
        }

        cspeed = 2.99752e5
        zsolar = 0.02

        # ------------------------------------ Load instantaneous burst model files ----------------------------------------
        # Read back metallicities from FITS headers (assume these are  correct!)
        n_met_all = 8
        zmod = np.zeros(n_met_all)

        hdulist = fits.open(os.path.join(settings['pipelinedir'],
                                         settings['burst_model_file']),
                            mode='denywrite', memmap=True,
                            do_not_scale_image_data=True)

        # Read metallicities from header fields
        for iz in range(n_met_all):
            zmod[iz] = hdulist[iz].header['Z']

        # Get header information for wavelength arrays
        n_pixmodel = hdulist[0].header['NAXIS1']
        n_burst = hdulist[0].header['NAXIS2']
        wl0 = hdulist[0].header['CRVAL1']
        dw = hdulist[0].header['CD1_1']
        burst_wl = np.arange(n_pixmodel) * dw + wl0

        # Create the model arrays using only the metallicities specified by the user in the settings
        n_met = len(settings['use_z'])

        burst_lib = np.zeros((n_pixmodel, n_burst, n_met))
        ssp_norms = np.zeros((n_burst, n_met))

        if 'burst' in settings['burst_model_file']:
            all_norms = hdulist[8].data[:]

        for imod in range(n_met):
            indx = np.array(np.where(zmod == settings['use_z'][imod])).squeeze()

            try:
                tmp = hdulist[np.int(indx)].data[:]
            except ValueError:
                sys.exit('ABORT -- requested model metallicity not found!')

            burst_lib[:, :, imod] = np.array(tmp).transpose()

            if 'burst' in settings['burst_model_file']:
                ssp_norms[:, imod] = all_norms[:, imod]

        ssp_ages = hdulist[n_met_all + 1].data[:]

        # model dispersion
        model_dispersion = 75.0
        settings['model_dispersion'] = model_dispersion

        settings['burst_lib'] = burst_lib
        settings['ssp_norms'] = ssp_norms
        settings['burst_wl'] = burst_wl
        settings['ssp_ages'] = ssp_ages
        settings['szval'] = settings['use_z']

        del hdulist

        self.settings = settings    
        
    def info(self, res):
        if res.get('spec', None) is not None:
            if hasattr(res['spec'], 'filename'):
                self.logger.info(f"Spectrum: {res['spec'].filename}")
        if res.get('success', None) is not None: 
            if res['success']:
                self.logger.info(f"Cont fit status: {res['status']}")
                self.logger.info(f"Cont Init Z: {res['init_z']:.5f}")
                self.logger.info(f"Cont Fit dV (km/s): {res['dv']:.2f}")
                self.logger.info(f"Cont Fit dZ: {res['dz']:.5f}")
                self.logger.info(f"Cont Fit Z: {res['z']:.5f}")
                self.logger.info(f"Cont Fit E(B-V): {res['ebv']:.2f}")
                self.logger.info(f"Cont Chi2: {res['chi2']:.2f}")
            else:
                self.logger.info(f"Cont fit status: {res['status']}")
                self.logger.info(f"Cont Init Z: {res['init_z']:.5f}")
                self.logger.info(f"Cont Fit dV (km/s): {res['dv']:.2f}")
                self.logger.info(f"Cont Fit dZ: {res['dz']:.5f}")
                self.logger.info(f"Cont Fit Z: {res['z']:.5f}")             

            
    def clean(self):
        """ clean all attributes, except the settings"""
        for attr in dir(self):
            if attr == 'settings':
                continue
            delattr(self, attr)
        
        
    def fit(self, spec, z, vdisp=80):
        """
        perform continuum fit on a mpdaf spectrum

        Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl)
        Translation of "fiber_continfit.pro" (part of the IDL PLATEFIT -
        contact: jarle@strw.leidenuniv.nl)

        Jan 18, 2019, Madusha:
        In the case NNLS fitting of the continuum is failed, settings returns best_sz = -99.0

        Apr 7, 2019, Roland:
        return cont,dz
        cont: mpdaf fitted continuum
        dz: fitted offset in z

        """
        cspeed = 2.99792E5
        
        # initialize result dict
        res = dict(spec=spec)

        flux_orig = spec.data
        err_orig = np.sqrt(spec.var)

        vacwl_orig = spec.wave.coord(medium='vacuum')

        # read information into the settings...
        self.settings['wavelength_range_for_fit'] = [vacwl_orig[0] / (1.0 + z), vacwl_orig[-1] / (1.0 + z)]

        tmp_logwl = np.log10(vacwl_orig)

        dw = np.min(np.abs(tmp_logwl - np.roll(tmp_logwl, 1)))
        if (dw == 0).any():
            sys.exit('ABORT: The wavelengths are identical somewhere!')

        # Shift wavelengths of the spectrum from air to vacuum. From now on, unless the wavelength is AIRWL,
        # the wavelengths below are in vacuum.

        xnew = np.arange(np.min(tmp_logwl), np.max(tmp_logwl), dw) # rebin in uniform step in log(vac)

        # mask bad or high std dev
        use = np.where((err_orig > 0.0) & (err_orig < 1.0E5))
        # interpolate over the new wavelengths
        ynew = np.interp(10**xnew, vacwl_orig[use], flux_orig[use])
        errnew = np.interp(10**xnew, vacwl_orig[use], err_orig[use])

        # final coordinates 
        vacwl = 10.0**xnew
        logwl = xnew
        flux = ynew
        err = errnew
        airwl = vactoair(vacwl)

        #restwl = airwl / (1.0 + z) # why in air ????
        restwl = vacwl / (1.0 + z)

        # ok is the mask
        ok = np.array(
            np.where((err > 0.0) & (np.isfinite(err**2) == True) & (err < 1.0E10) & (np.isfinite(flux) == True))).squeeze()

        # IMPORTANT NOTE:
        #       At this point the fluxes and errors need to be de-redden and adjust for (1+z) scaling. Data is NOT corrected
        #       for the forground extinction corrections. This functionality to be added later...
        #

        # scale by (1+z)
        flux = flux * (1.0 + z)
        err = err * (1.0 + z)

        # ----------------------------------------------------------------------------------------------------------------------
        #                                          Fit continuum using NNLS
        # ----------------------------------------------------------------------------------------------------------------------
        # set 'debug=True' for debug plots

        nsz = np.size(self.settings['szval'])

        npix = np.size(flux)
        continuum = np.zeros((npix, nsz))
        model_chi = np.zeros(nsz)
        model_contcoeff = np.zeros((np.shape(self.settings['burst_lib'])[1], nsz))
        model_library = np.zeros((npix, np.shape(self.settings['burst_lib'])[1], nsz))

        ebv = np.zeros_like(model_chi)

        best_continuum = np.zeros(npix) - 99.0
        best_modelChi = 99999.99
        best_modellib = np.zeros_like(model_library) - 99.0
        best_szval = -99.0
        best_ebv = -99.0

        redshift = np.array(z)
        flux_temp = np.array(flux[:])

        for isz in range(nsz):
            # Notice that the definition of chi squared is that returned by NNLS

            continuum[:, isz], settings_nnls = self.model_fit_nnls(logwl, flux, err, redshift, vdisp, isz,
                                                                   firstcall=True, debug=False)

            contcoefs = np.array(settings_nnls['params'][:])
            mean_chi2 = np.array(settings_nnls['mean'][:])

            if contcoefs[0] < -90:              
                self.logger.warning('Continnum fit failed, use constant')
                res['success'] = False
                res['status'] = 'Continnum fit failed, cste median used'
                res['dz'] = 0
                res['dv'] = 0
                res['ebv'] = 0
                res['z'] = z
                res['init_z'] = z
                res['chi2'] = 0

                cont = spec.clone()
                # cont is just the median over the wavelength range
                cont.data = np.ones_like(sp.data) * np.ma.median(sp.data)
                res['cont_spec'] = cont
                
                res['line_spec'] = spec - cont
                

                return res

            # Store the best-fit model results
            if mean_chi2 < best_modelChi:
                best_modelChi = mean_chi2
                best_modellib = np.array(settings_nnls['modellib'][:])
                ok_fit = np.array(settings_nnls['ok_fit'][:])
                best_szval = np.array(self.settings['szval'][isz])
                best_ebv = contcoefs[0]
                best_contCoefs = contcoefs
                best_continuum[:] = continuum[:, isz]

            # Store the model fit results correctly
            ebv[isz] = contcoefs[0]
            model_contcoeff[:, isz] = contcoefs[1:]
            model_chi[isz] = mean_chi2
            model_library[:, :, isz] = np.array(settings_nnls['modellib'][:])

            del contcoefs, mean_chi2, settings_nnls

        del flux
        flux = np.array(flux_temp[:])

        # fill result dict
        res['success'] = True
        res['status'] = 'Continuum fit successful'
        res['dz'] = best_szval
        res['dv'] = best_szval * cspeed
        res['ebv'] = best_ebv
        res['z'] = z + best_szval
        res['init_z'] = z
        res['chi2'] = best_modelChi[0]        
        res['ages'] = self.settings['ssp_ages'][np.array(np.where(best_contCoefs[1:] > 0)).squeeze()]
        res['weights'] = best_contCoefs[1:][np.array(np.where(best_contCoefs[1:] > 0)).squeeze()]

        cont = spec.clone()
        # rebin continuum in linear
        cont.data = np.interp(spec.wave.coord(), airwl, best_continuum)
        cont.data = cont.data / (1 + z)
        
        res['cont_spec'] = cont
        
        res['line_spec'] = spec - cont        
 
        return res


    def model_fit_nnls(self, logwl, flux, err, z, vdisp, modelsz, firstcall=None, debug=False):
        """
            Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl)
            Translation of "bc_model_fit_nnls.pro" (part of the IDL PLATEFIT - contact: jarle@strw.leidenuniv.nl)

        """
        nmodels = np.shape(self.settings['burst_lib'])[1]

        cspeed = 2.99792E5

        # de-redshift the data wave array and put in air
        npix = np.size(logwl)
        restwl = 10.0**logwl
        restwl = vactoair(restwl)
        restwl = restwl / (1.0 + z)

        # Interpolate models to match the data and convolve to velocity dispersion
        if firstcall is True:
            self.resample_model(logwl, z, vdisp, modelsz)

        modellib = self.settings['modellib']

        # -------------------------------------- Apply masks ---------------------------------------------------------------

        outside_model = np.where((restwl <= self.settings['wavelength_range_for_fit'][0]) |
                                 (restwl >= self.settings['wavelength_range_for_fit'][1]))

        quality = np.zeros(npix) + 1
        if np.size(outside_model) > 0:
            quality[outside_model] = 0

        # Grow masks a bit
        bad = np.where((np.isfinite(flux) == False) |
                       (np.isfinite(err) == False) | (err == 0))
        if np.size(bad) > 0:
            for i in range(-2, 2):
                quality[bad + i] = 0

        ok = np.array(np.where(quality == 1)).squeeze()

        if np.size(bad) > 0:
            for i in [-2, -1, 0, 1, 2]:
                quality[bad + i] = 0

        weight = np.zeros(npix)
        weight[ok] = 1.0 / err[ok]**2.0

        em = [3703.86,  # ; He I       1
              3726.03,  # ; [O II]     2
              3728.82,  # ; [O II]     3
              3750.15,  # ; H12        4
              3770.63,  # ; H11        5
              3797.90,  # ; H10        6
              3819.64,  # ; He I       7
              3835.38,  # ; H9         8
              3868.75,  # ; [Ne III]   9
              3889.05,  # ; H8        10
              3970.07,  # ; H-episilon 11
              4101.73,  # ; H-delta   12
              4026.21,  # ; He I      13
              4068.60,  # ; [S II]    14
              4340.46,  # ; H-gamma   15
              4363.21,  # ; [O III]
              4471.50,  # ; He I
              4861.33,  # ; H-beta    18
              4959.91,  # ; [O III]
              5006.84,  # ; [O III]
              5200.26,  # ; [N I]
              5875.67,  # ; He I
              5890.0,  # ; Na D (abs)
              5896.0,  # ; Na D (abs)
              6300.30,  # ; [O I]
              6312.40,  # ; [S III]
              6363.78,  # ; [O I]
              6548.04,  # ; [N II]
              6562.82,  # ; H-alpha   29
              6583.41,  # ; [N II]
              6678.15,  # ; He I
              6716.44,  # ; [S II]
              6730.81,  # ; [S II]
              7065.28,  # ; He I
              7135.78,  # ; [Ar III]
              7319.65,  # ; [O II]
              7330.16,  # ; [O II]
              7751.12,  # ; [Ar III]
              5577. / (1.0 + z)]  # ; night sky line

        # The full emission mask suit, mosltly relevant for the Antennae spectra
        # mask out emission lines, NaD, 5577 sky lines
        """
        em = [3703.86,  # He I       1
              3726.03,  # [O II]     2
              3728.82,  # [O II]     3
              3750.15,  # H12        4
              3770.63,  # H11        5
              3797.90,  # H10        6
              3819.64,  # He I       7
              3835.38,  # H9         8
              3868.75,  # [Ne III]   9
              3889.05,  # H8        10
              3970.07,  # H-epsilon 11
              4101.73,  # H-delta   12
              4026.21,  # He I      13
              4068.60,  # [S II]    14
              4340.46,  # H-gamma   15
              4363.21,  # [O III]   16
              4471.50,  # He I      17
              4861.33,  # H-beta    18
              4921.90,  #           19
              4959.91,  # [O III]   20
              5006.84,  # [O III]   21
              5016.60,  # HeI       22
              5200.26,  # [N I]     23
              5754.30,  # [N II]    24
              5875.67,  # He I      25
              5890.0,  # Na D (abs) 26
              5896.0,  # Na D (abs) 27
              6300.30,  # [O I]     28
              6312.40,  # [S III]   29
              6363.78,  # [O I]     30
              6548.04,  # [N II]    31
              6562.82,  # H-alpha   32
              6583.41,  # [N II]    33
              6678.15,  # He I      34
              6716.44,  # [S II]    35
              6730.81,  # [S II]    36
              7065.28,  # He I      37
              7135.78,  # [Ar III]  38
              7281.10,  # [HeI]     39
              7319.65,  # [O II]    40
              7330.16,  # [O II]    41
              7751.12,  # [Ar III]  42
              # 8046.00,
              8188.00,  #           43
              # 8216.60,
              # 8223.60,
              8398.00 / (1. + z),  # night sky line  44
              8392.60,  #           45
              8381.80 / (1. + z),  # night sky line  46
              8413.32,  # Pa19      47
              8437.96,  # Pa18      48
              8446.48,  # OI        49
              8467.26,  # Pa17      50
              8502.49,  # Pa16      51
              8545.38,  # Pa15      52
              8578.70,  # CIII      53
              8598.39,  # Pa14      54
              8665.02,  # Pa13      55
              8750.48,  # Pa12      56
              8862.79,  # Pa11      57
              9014.91,  # Pa10      58
              9068.90,  # SIII      59
              9229.02,  # Pa9       60
              5577. / (1. + z),  # night sky line
              6300. / (1. + z),  # night sky line
              6364. / (1. + z)]  # night sky line
        """

        # make mask width a multiple of the velocity dispersion
        if vdisp < 100.0:
            mask_width = 100.0
        elif vdisp > 500.0:
            mask_width = 500.0
        else:
            mask_width = vdisp

        mask_width = mask_width * 5.0
        mask_width = mask_width + np.zeros(np.size(em))

        for i in range(np.size(em)):
            voff = np.abs(restwl - em[i]) / em[i] * cspeed
            maskout = np.where(voff < mask_width[i])

            if np.size(maskout) > 0:
                quality[maskout] = 0

        ok = np.array(np.where(quality == 1)).squeeze()
        not_ok = np.array(np.where(quality == 0)).squeeze()

        # ----------------------------------- Call the fitting routine -----------------------------------------------------

        settings_nnls = {}  # Declare the 'settings_nnls' structure which will hold the NNLS outputs
        settings_nnls = self.fit_burst_nnls(flux, restwl, 1.0 / np.sqrt(weight), ok, settings_nnls)

        fitcoefs = settings_nnls['params']
        yfit = self.model_combine(restwl, fitcoefs, settings_nnls)

        settings_nnls['ok_fit'] = ok
        settings_nnls['not_ok_fit'] = not_ok
        settings_nnls['modellib'] = modellib

        # ------------------------------------------------------------------------------------------------------------------
        # If debugging plot spectrum, best fit, and individual stellar components
        # ------------------------------------------------------------------------------------------------------------------
        if debug is True:
            ymax = np.max(yfit) * 1.1

            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(restwl[ok], flux[ok], 'k-')

            plt.plot(restwl, flux - 400., 'y-')
            plt.plot(restwl, yfit, 'r-')

            for i in range(nmodels):
                yi = fitcoefs[i + 1] * modellib[:, i] * \
                    np.exp(-fitcoefs[0] * (restwl / 5500.0)**(-0.7))
                plt.plot(restwl, yi, 'b-')

            plt.xlim([4600., 9300.])
            plt.ylim([0.0, ymax])
            plt.show()

        return yfit, settings_nnls

    def resample_model(self, logwl, z, vdisp, modelsz):
        """
        Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl)
        Translation of "resample_model.pro" (part of the IDL PLATEFIT -
        contact: jarle@strw.leidenuniv.nl)

        """
        nmodels = np.shape(self.settings['burst_lib'])[1]

        # ------------------------ redshift model wave array, and put in vaccum --------------------------------------------
        obs_burst_wl = self.settings['burst_wl']
        obs_burst_wl = obs_burst_wl * (1.0 + z)
        obs_burst_wl = airtovac(obs_burst_wl)

        burst_lib = self.settings['burst_lib']

        # de-redshift data wave array, and put in air only for comparison to burst_wl
        npix = np.size(logwl)
        restwl = 10.0**logwl

        # Convert wavelength from vacuum to air
        restwl = vactoair(restwl)  # The wavelengths in the linepar are in air
        restwl = restwl / (1.0 + z)

        # Convert wavelength in air to wavelength
        # in vacuum
        # restwl = pyasl.airtovac(restwl)

        # -------------------- Interpolate models to match data & convolve to velocity dispersion --------------------------
        # This is the resolution of the templates
        data_disp = self.settings['model_dispersion']

        cspeed = 2.99792E5
        loglamtov = cspeed * np.log(10.0)

        dw = logwl[1] - logwl[0]

        # Figure out convolution 'sigma' in units of pixels, being sure to deconvolve the template resolution first
        if vdisp <= data_disp:
            sigma_pix = 50.0 / loglamtov / dw
            self.logger.warning('vdisp < the dispersion of the templates. sigma_pix is set to 50.0km/s')
        else:
            vdisp_add = np.sqrt(vdisp**2 - data_disp**2)  # Deconvolve template resolution
            sigma_pix = vdisp_add / loglamtov / dw

        # Interpolate reshifted models to the same wavelength grid as the data (in log-lambda) and then convolve to the
        # data velocity dispersion
        custom_lib = np.zeros((npix, nmodels))
        temp_wl = self.settings['burst_wl']

        # Another way of doing the convolution (using scipy.convolve). But I am sticking with the ppxf version.
        # gauss_kernel = Gaussian1DKernel(sigma_pix)

        for i in range(nmodels):
            burst = np.interp(logwl, np.log10(obs_burst_wl), burst_lib[:, i, modelsz])
            # burst = np.interp(logwl, np.log10(obs_burst_wl), burst_lib[:, i, modelsz])

            # Smooth the template with a Gaussian filter.
            #custom_lib[0:, i] = ppxf_util.gaussian_filter1d(burst, sigma_pix)
            # use the astropy convolution (gives better results on the edge)
            gauss = Gaussian1DKernel(stddev=sigma_pix)
            custom_lib[0:, i] = convolve(burst, gauss)
            # custom_lib[0:, i] = sp.ndimage.filters.gaussian_filter1d(burst, sigma_pix, order=0)
            # custom_lib[0:, i] = convolve(burst, gauss_kernel)

        # -------------------------- set regions outside of the mode to zero -----------------------------------------------
        outside_model = np.where((restwl <= np.min(temp_wl)) | (restwl >= np.max(temp_wl)))
        if np.size(outside_model) > 0:
            custom_lib[outside_model, :] = 0.0

        # -------------------------- load in to the settings ---------------------------------------------------------------
        self.settings['modellib'] = custom_lib
        self.settings['burst_wllib'] = temp_wl

        return

    def fit_burst_nnls(self, flux, wavelength, dflux, ok, settings_nnls):
        """
        Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl)
        Translation of "fit_burst_nnls.pro" (part of the IDL PLATEFIT -
        contact: jarle@strw.leidenuniv.nl)

        """

        modellibrary = self.settings['modellib']

        # Weights the flux by the error
        b = np.array(flux[ok]) / np.array(dflux[ok])
        lam = np.array(wavelength[ok]).squeeze()

        dims = np.shape(modellibrary)
        n = dims[1]
        m = np.size(ok)

        # Various other input/output parameters
        params = np.zeros(n + 1)  # N models + EBV. Where params[0] will be EBV
        nparams = n + 1

        # Weight the model library by the errors too
        a = np.zeros((m, n))
        for i in range(n):
            temp = np.array(modellibrary[ok, i]) / np.array(dflux[ok])
            a[:, i] = np.array(temp).squeeze()

        # Adjust the input parameters to match the shape and precision
        a = np.array(a).transpose().reshape(1, np.size(a))
        a = np.array(a).squeeze()
        nxm = m
        nmodel = n
        lam = np.array(lam).squeeze()
        flux = b
        params = params
        nparams = nparams
        nm = np.size(a)
        mean = np.zeros(1)  # one of the inputs/outputs of fortran. NOTE the way the variable is declared
        sigma = np.zeros(1)  # one of the inputs/outputs of fortran

        # ----------------------------- Call the external fortran routine --------------------------------------------------
        # fit_continuum1test(np.long(nparams), np.int(nxm), np.long(nmodel), np.long(nm), np.double(lam), np.double(flux),
        #                    np.double(a), np.double(params), np.double(mean), np.double(sigma))
        # fit_continuum1test(lam, flux, params, nxm, nparams)

        # print(nxm, nm, nparams, nmodel)
        # lambda, flux, models, parameters, mean, sigma, nmodels, ok, size(a), nparams
        fit_continuum1(lam, np.double(flux), np.double(a), np.double(params),
                       mean, sigma, nmodel, nxm, nm, nparams)

        # print('parameters=', params)
        # print(mean, sigma)

        # Write the results in to the 'settings_nnls' structure
        settings_nnls['params'] = params
        settings_nnls['mean'] = mean
        settings_nnls['sigma'] = sigma

        return settings_nnls

    def model_combine(self, x, a, settings_nnls, good_data=None, ssp_ages=None,
                      individual=False, correct=None):
        """
        Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl)
        Translation of "bc_model_combine.pro" (part of the IDL PLATEFIT -
        contact: jarle@strw.leidenuniv.nl)

        """
        #
        # At the moment - CORRECT is for using the same as the Fortran code does,
        # namely no reddening difference between young and old population
        #

        modellib = self.settings['modellib']
        sz = np.shape(modellib)
        nxlib = sz[0]
        nmodels = sz[1]

        if good_data is None:
            good_data = np.arange(nxlib)
        ngood = np.size(good_data)

        if ssp_ages is None:
            ssp_ages = np.zeros(nmodels)

        # ------------------------------------------------------------------------------------------------------------------
        # Create a linear combination of templates
        # Redden using the Charlot & Fall law with the time dependent normalisation
        # F_obs = F_int * exp(-Tau_V * (lambda / 5500 A)^-0.7)
        # ------------------------------------------------------------------------------------------------------------------

        # Old method as indicated in Jarle's code
        # y = np.matmul(modellib[good_data, :], a[1:, :])
        # klam = (x / 5500.0)**(-0.7)
        # e_tau_lam = np.exp(-a[0] * klam)
        # y = y * e_tau_lam

        y = 0.0
        klam = (x / 5500.0)**(-0.7)

        if individual is True:
            individuals = modellib * 0.0

        # ----------------------The original Bug report from 'bc_model_combine.pro'-----------------------------------------
        # Bug: 21/07/2013 - this is in principle fine, but the fitting routine does not treat this age cut properly.
        #                   Thus the scaling of the youngest component is incorrect. This is probably not a big deal
        #                   for the emission line fitting because this incorporates a smooth correction to the fit
        #                   but still it should be fixed.
        # ------------------------------------------------------------------------------------------------------------------

        for i in range(nmodels):
            if ssp_ages[i] < 1.0E7:
                norm = 1.0
            else:
                norm = 1.0 / 3.0

            # CORRECT routine!
            if correct is not None: norm = 1.0

            tmp = modellib[good_data, i] * a[i + 1] * np.exp(-a[0] * norm * klam)
            y = y + tmp

            if individual is True:
                individuals[good_data, i] = tmp

        # ------------------------------------------------------------------------------------------------------------------
        # Calculate the dy/da partial derivatives
        # Not correct any longer -- redo in using again
        # The commneted out IDL routine from 'bc_model_combine.pro' is written below
        # ------------------------------------------------------------------------------------------------------------------
        #
        # if n_params() gt 2 then begin
        #     pder = dblarr(ngood, nmodels+1)
        #     pder[*, 0] = -y * klam
        #     for i=0, nmodels - 1 do pder[*,i+1] = modellib[good_data, i] * e_tau_lam
        # endif
        #
        # ------------------------------------------------------------------------------------------------------------------

        # Write to the settings_nnls structure
        if individual is True:
            settings_nnls['individuals'] = individuals

        return np.array(y)
    