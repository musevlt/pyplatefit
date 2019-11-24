Tutorial
========

This tutorial show how to use pyplatefit to perform continuum and emission line
fit.

.. _basic:

Basic usage
+++++++++++

.. code::

   from pyplatefit import fit_spec
   from mpdaf.obj import Spectrum
   sp = Spectrum('test_data/udf10_00002.fits')
   z = 0.41892
   res = fit_spec(sp, z)
   
::

	[DEBUG] Performing continuum and line fitting
	[DEBUG] Getting lines from get_emlines...
	[DEBUG] 21.3 % of the spectrum is used for fitting.
	[DEBUG] Found 2 non resonnant line families to fit
	[DEBUG] Performing fitting of family balmer
	[DEBUG] LSQ Fitting of 9 lines
	[DEBUG] added 9 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 85 iterations, redChi2 = 249.367
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Performing fitting of family forbidden
	[DEBUG] LSQ Fitting of 13 lines
	[DEBUG] added 13 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 148 iterations, redChi2 = 255.486
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Found 0 resonnant line families to fit

By default the program perform independant fit for each line family. In this case, the
fit was performed for the Balmer and Forbidden families with respectively 9 and 13
emission lines.
A unique velocity offset (and corresponding redshift) an velocity dispersion is fitted
for all the lines of the same family. See section :ref:`emlines`. for more details.




.. note::

   It is possible to use directly the ``sp.fit_lines()`` method to get the same
   results without the need to import pyplatefit.
   
.. code::

   from mpdaf.obj import Spectrum
   sp = Spectrum('test_data/udf10_00002.fits')
   z = 0.41892
   res = sp.fit_lines(z)
   
The ``res`` dictionary contains all the fit results. Consult the API documentation
for the full description at `pyplatefit.fit_lines`.

The astropy tables lines and ztable are indexed respectively by LINE (line name) and FAMILY (line family name).
For example to access the result of lines for LYALPHA, one can use :

.. code::

   row = res['lines'].loc['LYALPHA']

Let's first display a summary of the fit results by family:

.. code::

   res['ztable'].pprint_all()
   
::

	  FAMILY   VEL  VEL_ERR    Z     Z_ERR    Z_INIT VDISP VDISP_ERR SNRMAX SNRSUM SNRSUM_CLIPPED  NL NL_CLIPPED NFEV RCHI2 
	--------- ----- ------- ------- -------- ------- ----- --------- ------ ------ -------------- --- ---------- ---- ------
	   balmer 82.14    3.86 0.41919 1.29e-05 0.41892 66.11      4.26  20.63  13.28          12.97   9          5   85 249.37
	forbidden 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  13.94  16.84          20.69  13          6  148 255.49

As we can see the velocity offset is slightly different for the Balmer and Forbidden
lines. The columns SNRMAX, SNRSUM and SNRSUM_CLIPPED are useful information to decide
on the relevance of the fit.

.. Note::

   Redshift are given in vacuum and all given values are given in rest frame, except
   if specified. The velocity dispersion is the intrinsic value, corrected by the
   instrumental LSF, except if the option ``lsf=False`` is used.

In this specific case the two families given similar results and it can be useful to 
fit all lines simultaneously.

.. code::

   res = sp.fit_lines(z, fit_all=True)
   res['ztable'].pprint_all()

::

	[DEBUG] Performing continuum and line fitting
	[DEBUG] Getting lines from get_emlines...
	[DEBUG] 21.3 % of the spectrum is used for fitting.
	[DEBUG] Performing fitting of all expect Lya lines together
	[DEBUG] LSQ Fitting of 22 lines
	[DEBUG] added 22 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 358 iterations, redChi2 = 19.921
	[DEBUG] Saving results to tablines and ztab
	
	FAMILY  VEL  VEL_ERR    Z     Z_ERR    Z_INIT VDISP VDISP_ERR SNRMAX SNRSUM SNRSUM_CLIPPED  NL NL_CLIPPED NFEV RCHI2
	------ ----- ------- ------- -------- ------- ----- --------- ------ ------ -------------- --- ---------- ---- -----
	   all 85.88    0.88 0.41921 2.92e-06 0.41892 65.86      0.98  75.39  62.12          69.13  22         16  358 19.92	

The next step is to visualize the fit quality.

.. code::

   import matplotlib.pyplot as plt
   from pyplatefit import plot_fit
   fig,ax = plt.subplots(1,3, figsize=(15,5))
   plot_fit(ax[0], res, iden=False)
   plot_fit(ax[1], res, line='HBETA')
   plot_fit(ax[2], res, line='HBETA', line_only=True, start=True)
   plt.show()
   
.. image:: images/high_fig1.png

One can see on the left, the continuum and full spectrum fit, on the center a zoom
on the Hbeta line and on the right the line fit performed on the continuum subtracted
spectrum and the initial solution of the fit (in blue).


The individual line information is given in the ``lines`` table. 
   
.. code::

   res['lines'].pprint_all()
   
will write the following:

::

	  FAMILY     LINE   LBDA_REST  DNAME   VEL  VEL_ERR    Z     Z_ERR    Z_INIT VDISP VDISP_ERR VDINST   FLUX   FLUX_ERR  SNR  SKEW SKEW_ERR LBDA_OBS PEAK_OBS LBDA_LEFT LBDA_RIGHT FWHM_OBS RCHI2   EQW   EQW_ERR CONT_OBS   CONT  CONT_ERR
	--------- --------- --------- ------- ----- ------- ------- -------- ------- ----- --------- ------ -------- -------- ----- ---- -------- -------- -------- --------- ---------- -------- ------ ------ ------- -------- ------- --------
	forbidden   NEV3427   3426.85     Neᴠ 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  70.31     0.08   496.73  0.00   --       --  4863.48     0.02   4861.63    4865.33     3.70 255.49  -0.00    0.61   574.57  815.27    30.12
	forbidden   OII3727   3727.09    None 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  62.02  4340.83   426.08 10.19   --       --  5289.59  1078.09   5287.70    5291.48     3.78 255.49  -5.19    0.53   589.73  836.78    41.88
	forbidden   OII3729   3729.88   [Oɪɪ] 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  61.95  6065.89   435.24 13.94   --       --  5293.55  1506.17   5291.66    5295.44     3.78 255.49  -7.20    0.55   593.92  842.73    42.22
	   balmer       H11   3771.70     H11 82.14    3.86 0.41919 1.29e-05 0.41892 66.11      4.26  60.94   196.53   395.95  0.50   --       --  5352.77    48.84   5350.88    5354.66     3.78 249.37  -0.20    0.41   678.39  962.58    46.23
	   balmer       H10   3798.98     H10 82.14    3.86 0.41919 1.29e-05 0.41892 66.11      4.26  60.29   323.31   384.21  0.84   --       --  5391.49    80.15   5389.59    5393.38     3.79 249.37  -0.33    0.39   700.12  993.41    41.11
	   balmer        H9   3836.47      H9 82.14    3.86 0.41919 1.29e-05 0.41892 66.11      4.26  59.43   573.32   380.70  1.51   --       --  5444.70   141.67   5442.79    5446.60     3.80 249.37  -0.54    0.36   744.95 1057.03    40.19
	forbidden NEIII3870   3870.16 [Neɪɪɪ] 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  58.67   401.94   386.22  1.04   --       --  5492.64    98.57   5490.73    5494.56     3.83 255.49  -0.38    0.36   752.97 1068.41    38.28
	forbidden   HEI3890   3889.73    None 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  58.24  1343.18   392.77  3.42   --       --  5520.42   328.79   5518.50    5522.33     3.84 255.49  -1.23    0.37   771.42 1094.59    63.25
	   balmer        H8   3890.15      H8 82.14    3.86 0.41919 1.29e-05 0.41892 66.11      4.26  58.23  1302.18   386.56  3.37   --       --  5520.88   320.20   5518.97    5522.79     3.82 249.37  -1.19    0.36   770.81 1093.72    63.27
	forbidden NEIII3967   3968.91    None 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  56.55   759.29   386.03  1.97   --       --  5632.79   184.45   5630.86    5634.72     3.87 255.49  -0.67    0.34   802.90 1139.25    59.16
	   balmer  HEPSILON   3971.20      Hε 82.14    3.86 0.41919 1.29e-05 0.41892 66.11      4.26  56.51  1107.87   378.29  2.93   --       --  5635.90   270.33   5633.98    5637.83     3.85 249.37  -0.97    0.34   805.59 1143.07    59.02
	   balmer    HDELTA   4102.89      Hδ 82.14    3.86 0.41919 1.29e-05 0.41892 66.11      4.26  53.91  2051.52   379.23  5.41   --       --  5822.80   493.96   5820.85    5824.75     3.90 249.37  -1.75    0.33   825.06 1170.69    35.49
	   balmer    HGAMMA   4341.68      Hγ 82.14    3.86 0.41919 1.29e-05 0.41892 66.11      4.26  49.79  3648.03   348.88 10.46   --       --  6161.69   855.57   6159.68    6163.69     4.01 249.37  -3.28    0.32   784.02 1112.46    36.08
	forbidden  OIII4364   4364.44    None 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  49.43    27.45   346.28  0.08   --       --  6194.14     6.39   6192.12    6196.16     4.04 255.49  -0.02    0.31   798.74 1133.35    36.72
	   balmer     HBETA   4862.68      Hβ 82.14    3.86 0.41919 1.29e-05 0.41892 66.11      4.26  42.92  8568.01   415.39 20.63   --       --  6901.09  1883.80   6898.95    6903.22     4.27 249.37  -7.86    0.40   768.17 1089.98    30.89
	forbidden  OIII4960   4960.30    None 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  41.90   654.24   265.55  2.46   --       --  7039.80   141.19   7037.62    7041.98     4.35 255.49  -0.59    0.24   778.55 1104.70    23.15
	forbidden  OIII5008   5008.24  [Oɪɪɪ] 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  41.43  2215.83   272.60  8.13   --       --  7107.84   475.12   7105.65    7110.03     4.38 255.49  -2.03    0.25   770.43 1093.18    24.17
	forbidden   HEI5877   5877.25    None 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  35.41   907.92   459.28  1.98   --       --  8341.16   172.50   8338.69    8343.63     4.94 255.49  -0.88    0.45   723.61 1026.75    48.67
	forbidden    OI6302   6302.05    [Oɪ] 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  33.85   723.60   953.43  0.76   --       --  8944.05   129.45   8941.42    8946.67     5.25 255.49  -0.75    1.00   679.34  963.93    81.74
	forbidden   NII6550   6549.85    None 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  33.26  4502.11   691.86  6.51   --       --  9295.73   777.70   9293.01    9298.45     5.44 255.49  -4.61    0.74   688.85  977.43    65.37
	   balmer    HALPHA   6564.61      Hα 82.14    3.86 0.41919 1.29e-05 0.41892 66.11      4.26  33.23 23689.35  2927.65  8.09   --       --  9316.45  4110.00   9313.75    9319.16     5.42 249.37 -24.06    3.27   693.90  984.59   114.60
	forbidden   NII6585   6585.28    None 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  33.19 11604.35  1017.52 11.40   --       --  9346.02  1994.57   9343.28    9348.75     5.47 255.49 -11.65    1.45   701.99  996.06   292.68

For the detail of all columns consult the `pyplatefit.fit_spec` informations. 

.. _doublet:

Emission lines doublet
++++++++++++++++++++++

Lines doublet are always fitted together. For some doublet, namely [OII] and [CIII], 
it is possible to constrain the line ratio in a given interval. This is done with
the option ``use_line_ratios`` in `pyplatefit.fit_spec`. The line ratios have some
default values (0.6-1.2 for CIII and 1.0-2.0 for OII), which can be overriden 
in the ``linepars`` argument optional dictionary. See an example below:

.. code::

    ratio = [("OII3727", "OII3729", 1.0, 1.5)]
    res = fit_spec(sp, z, use_line_ratios=True, linepars={'line_ratios':ratio})

Note that imposing constrain on line ratios can sometimes prevent lmfit LSQ fitting
to report errors. If a good estimate of SNR is important, it is probably better not 
to activate this option. Alternatively using the ``emcee`` option is possible. See 
section :ref:`faint`.


.. _resonant:

Resonant emission lines
++++++++++++++++++++++++

Resonant emission lines can have a different velocity offset from non-resonant lines
and need to be fitted individually (or by doublet). The list of resonant lines 
is defined in :ref:`emlines`.

When fitting a resonant line, the family name is the name of the line in uppercase, or
the name of the first line in the case of a doublet.

.. code::

   from mpdaf.obj import Spectrum
   sp = Spectrum('test_data/udf10_00056.fits')
   z = 1.30604
   res = sp.fit_lines(z)
   res['ztable'].pprint_all()

::

	[DEBUG] Performing continuum and line fitting
	[DEBUG] Getting lines from get_emlines...
	[DEBUG] 17.2 % of the spectrum is used for fitting.
	[DEBUG] Found 2 non resonnant line families to fit
	[DEBUG] Performing fitting of family balmer
	[DEBUG] LSQ Fitting of 5 lines
	[DEBUG] added 5 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 41 iterations, redChi2 = 13.448
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Performing fitting of family forbidden
	[DEBUG] LSQ Fitting of 8 lines
	[DEBUG] added 8 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 137 iterations, redChi2 = 1.854
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Found 2 resonnant line families to fit
	[DEBUG] Performing fitting of family cii2326
	[DEBUG] LSQ Fitting of ['CII2326']
	[DEBUG] added 1 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 61 iterations, redChi2 = 13.558
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Performing fitting of family mgii2796
	[DEBUG] LSQ Fitting of ['MGII2796', 'MGII2803']
	[DEBUG] added 2 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 27 iterations, redChi2 = 12.644
	[DEBUG] Saving results to tablines and ztab

	  FAMILY   VEL   VEL_ERR    Z     Z_ERR    Z_INIT VDISP  VDISP_ERR SNRMAX SNRSUM SNRSUM_CLIPPED  NL NL_CLIPPED NFEV RCHI2
	--------- ------ ------- ------- -------- ------- ------ --------- ------ ------ -------------- --- ---------- ---- -----
	   balmer  41.75   50.85 1.30618 1.70e-04 1.30604 100.67     51.24   1.61   2.51             --   5          0   41 13.45
	forbidden  78.95    1.32 1.30630 4.40e-06 1.30604  41.17      1.61  44.90  30.02          42.02   8          4  137  1.85
	  cii2326 229.87  603.73 1.30681 2.01e-03 1.30604 299.97    495.62   0.58   0.58             --   1          0   61 13.56
	 mgii2796 109.40   14.19 1.30640 4.73e-05 1.30604  50.59     19.29   5.38   5.09           5.38   2          1   27 12.64   

Note that the resonant lines will be fitted with all other lines when the option 
``fit_all`` is activated.
   

.. _lya:

Lyman alpha emission line 
+++++++++++++++++++++++++

The lyman alpha line is a resonant line with an asymetric shape. It is then always
fitted independently (even when the option ``fit_all`` is activated). While other lines
are modelled as Gaussian, we use the skew normal distribution describe
eg in `wikipedia <https://en.wikipedia.org/wiki/Skew_normal_distribution>`_.
The skewness parameter used in the model is named SKEW in the ``lines`` table.

.. code::

   sp = Spectrum('test_data/udf10_00053.fits')
   z = 4.77666
   res = fit_spec(sp, z, fit_all=True)
   res['ztable'].pprint_all()
   
::

	[DEBUG] Performing continuum and line fitting
	[DEBUG] Getting lines from get_emlines...
	[DEBUG] 6.8 % of the spectrum is used for fitting.
	[DEBUG] LSQ Fitting of Lya
	[DEBUG] Computed Lya init velocity offset: 82.15
	[DEBUG] added 1 asymetric gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 49 iterations, redChi2 = 2.976
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Performing fitting of all expect Lya lines together
	[DEBUG] LSQ Fitting of 4 lines
	[DEBUG] added 4 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 71 iterations, redChi2 = 274.976
	[DEBUG] Saving results to tablines and ztab

	 FAMILY  VEL   VEL_ERR    Z     Z_ERR    Z_INIT VDISP  VDISP_ERR SNRMAX SNRSUM SNRSUM_CLIPPED  NL NL_CLIPPED NFEV RCHI2
	------- ------ ------- ------- -------- ------- ------ --------- ------ ------ -------------- --- ---------- ---- ------
	lyalpha  86.40    1.32 4.77695 4.39e-06 4.77666 284.52      3.25 119.36 119.36         119.36   1          1   49   2.98
		all -22.28 1081.43 4.77659 3.61e-03 4.77666 211.46   1114.39   0.20   0.12             --   4          0   71 274.98


.. code::

   fig,ax = plt.subplots(1,1) 
   res['line_spec'].plot(ax=ax)
   res['line_fit'].plot(ax=ax, color='r')
   ax.set_xlim(7000,7060);
   plt.show()
   
.. image:: images/high_fig2.png

.. code::

	tab = res['lines']
	tab.add_index('LINE')
	tab.loc['LYALPHA']
	tab.loc['LYALPHA'][['SKEW','SKEW_ERR']]
	
::

	  SKEW  SKEW_ERR
	float64 float64
	------- --------
	   7.25     0.37


In this highly asymmetric case the skewness parameter reach 7.25.

   
Double peaked Lyman alpha emission line 
+++++++++++++++++++++++++++++++++++++++

When the lyman alpha line is double peaked one can use the option ``dble_lyafit`` to perform
the simultaneous fit of the the two lines. The model is the sum of two asymetric gaussian. 
The input and returned redshift refer to the midpoint of the two lines.

.. code::

   sp = Spectrum('test_data/udf10_00106.fits')
   z = 3.27554
   res = fit_spec(sp, z, lines=['LYALPHA'], dble_lyafit=True, find_lya_vel_offset=False)
   res['lines'].loc['LYALPHA']['LINE','Z','SEP','VEL','VDISP','FLUX','SKEW','LBDA_OBS']
   
::

          LINE     Z      SEP     VEL    VDISP    FLUX    SKEW  LBDA_OBS
         str20  float64 float64 float64 float64 float64 float64 float64 
        ------- ------- ------- ------- ------- ------- ------- --------
        LYALPHA 3.27603  515.76   34.15  194.34  680.35   -2.78  5190.42
        LYALPHA 3.27603  515.76   34.15  307.69 1080.50    4.05  5203.69

The fitting parameters are : 

   - VEL, the rest frame velocity offset in km/s
   - SEP, the rest frame peak separation in km/s
   - VDISP, the rest frame velocity dispersion (km/s) of each component
   - FLUX, the flux of each component
   - SKEW, the skewness parameter of each component

Note that it is better to deactivate the automatic search of lya peak (``find_lya_vel_offset=False``).
The fit can be displayed with ``plot_fit``. 

.. code::

   fig,ax = plt.subplots(1,1) 
   plot_fit(ax, res, line='LYALPHA', line_only=True)
   plt.show()
   
.. image:: images/high_fig3.png


.. _faint:

Working with faint emission lines
+++++++++++++++++++++++++++++++++

Faint emission lines can be challenging for least-square fitting. Even if the line flux are 
constrain to be positive, the solution returned by lmfit may nit be very accurate
and the errors will probably be largely underestimated. 

In this case it is recommended to use the option ``emcee=True``.
After a first least-square fit a second minimisation is performed using Bayesian 
sampling of the posterior distribution with the EMCEE 
routine of ``lmfit``. This will give a better estimate of errors, but note that it is
computationally expensive.

.. code::

   sp = Spectrum('test_data/udf10_00723.fits')
   z = 3.18817
   res = fit_spec(sp, z)
   res['ztable'].pprint_all()
   res['lines'][['FAMILY','LINE','FLUX','FLUX_ERR','SNR']].pprint_all()
   
   
::

	[DEBUG] Performing continuum and line fitting
	[DEBUG] Getting lines from get_emlines...
	[DEBUG] 10.0 % of the spectrum is used for fitting.
	[DEBUG] LSQ Fitting of Lya
	[DEBUG] Computed Lya init velocity offset: 72.80
	[DEBUG] added 1 asymetric gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 54 iterations, redChi2 = 0.324
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Found 1 non resonnant line families to fit
	[DEBUG] Performing fitting of family forbidden
	[DEBUG] LSQ Fitting of 9 lines
	[DEBUG] added 9 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Too many function calls (max set to 1000)!  Use: minimize(func, params, ..., maxfev=NNN)or set leastsq_kws['maxfev']  to increase this maximum. Could not estimate error-bars. after 1006 iterations, redChi2 = 0.394
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Found 0 resonnant line families to fit
	
	  FAMILY   VEL   VEL_ERR    Z     Z_ERR    Z_INIT VDISP  VDISP_ERR SNRMAX SNRSUM SNRSUM_CLIPPED  NL NL_CLIPPED NFEV RCHI2
	--------- ------ ------- ------- -------- ------- ------ --------- ------ ------ -------------- --- ---------- ---- -----
	  lyalpha  37.03   22.52 3.18829 7.51e-05 3.18817 263.94     50.87   7.13   7.13           7.13   1          1   54  0.32
	forbidden -27.48      -- 3.18808       -- 3.18817 120.20        --     --     --             --  --         -- 1006  0.39
	
	  FAMILY    LINE    FLUX  FLUX_ERR SNR
	--------- -------- ------ -------- ----
	  lyalpha  LYALPHA 117.54    16.48 7.13
	forbidden  NeV1238  15.54       --   --
	forbidden  NeV1243   0.00       --   --
	forbidden  CIV1548   0.00       --   --
	forbidden  CIV1551  12.19       --   --
	forbidden HEII1640   0.01       --   --
	forbidden OIII1660  10.74       --   --
	forbidden OIII1666   2.09       --   --
	forbidden CIII1907  20.23       --   --
	forbidden CIII1909   9.59       --   --	
   	
In this case, the lyman alpha line was successfully fitted, but not the faint forbidden 
lines, resulting in the absence of information of the SNR. If we now use the 
``emcee`` option, we obtain:
 
.. code::

   res = fit_spec(sp, z, emcee=True)
   res['ztable'].pprint_all()
   res['lines'][['FAMILY','LINE','FLUX','FLUX_ERR','SNR']].pprint_all()
   
   
:: 
 
	[DEBUG] Performing continuum and line fitting
	[DEBUG] Getting lines from get_emlines...
	[DEBUG] 10.0 % of the spectrum is used for fitting.
	[DEBUG] LSQ Fitting of Lya
	[DEBUG] Computed Lya init velocity offset: 72.80
	[DEBUG] added 1 asymetric gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 54 iterations, redChi2 = 0.324
	[DEBUG] Error estimation using EMCEE with nsteps: 1000 nwalkers: 12 burn: 20
	[DEBUG] End EMCEE after 12000 iterations, redChi2 = 0.325
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Found 1 non resonnant line families to fit
	[DEBUG] Performing fitting of family forbidden
	[DEBUG] LSQ Fitting of 9 lines
	[DEBUG] added 9 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Too many function calls (max set to 1000)!  Use: minimize(func, params, ..., maxfev=NNN)or set leastsq_kws['maxfev']  to increase this maximum. Could not estimate error-bars. after 1006 iterations, redChi2 = 0.394
	[DEBUG] Error estimation using EMCEE with nsteps: 1000 nwalkers: 34 burn: 20
	[DEBUG] End EMCEE after 34000 iterations, redChi2 = 0.393
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Found 0 resonnant line families to fit

	  FAMILY   VEL   VEL_ERR    Z     Z_ERR    Z_INIT VDISP  VDISP_ERR SNRMAX SNRSUM SNRSUM_CLIPPED  NL NL_CLIPPED  NFEV RCHI2
	--------- ------ ------- ------- -------- ------- ------ --------- ------ ------ -------------- --- ---------- ----- -----
	  lyalpha  86.40   61.11 3.18846 2.04e-04 3.18817 265.37     92.75   4.16   4.16           4.16   1          1 12000  0.32
	forbidden -28.27  104.34 3.18808 3.48e-04 3.18817 214.21     84.76   1.17   2.31             --   9          0 34000  0.39 
 
	  FAMILY    LINE    FLUX  FLUX_ERR SNR
	--------- -------- ------ -------- ----
	  lyalpha  LYALPHA 117.27    28.21 4.16
	forbidden  NeV1238  26.81    27.95 0.96
	forbidden  NeV1243   0.00     0.00 0.52
	forbidden  CIV1548   0.00     0.00 0.33
	forbidden  CIV1551  16.15    15.98 1.01
	forbidden HEII1640   0.02     0.04 0.38
	forbidden OIII1660  17.62    16.12 1.09
	forbidden OIII1666   5.89     8.69 0.68
	forbidden CIII1907  21.59    18.47 1.17
	forbidden CIII1909  15.24    17.08 0.89


We now have a good estimate of the SNR for all faint lines. Note also that the previous
estimate of the SNR with LSQ has reduced from 7.13 to the more realistic value of 4.16.

.. _contfit:

Continuum fit
+++++++++++++

The continuum fit assume that the input redshift is good enough. If this is not the
case, the continuum fit will not be accurate, which will then impact the emission 
line fit after continuum subtraction. In this case there is an option ``ziter`` 
which force a second continuum fit once the redshift has been refined by the
first iteration.

.. code::

	sp = Spectrum('test_data/udf10_00002.fits')
	z = 0.418
	res = fit_spec(sp, z, ziter=True)
	
::

	[DEBUG] Performing continuum and line fitting
	[DEBUG] Performing a first quick fit to refine the input redshift
	[DEBUG] Getting lines from get_emlines...
	[DEBUG] 21.3 % of the spectrum is used for fitting.
	[DEBUG] Performing fitting of all expect Lya lines together
	[DEBUG] LSQ Fitting of 22 lines
	[DEBUG] added 22 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 562 iterations, redChi2 = 55.762
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Computed velocity offset 280.7 km/s
	[DEBUG] Getting lines from get_emlines...
	[DEBUG] 21.2 % of the spectrum is used for fitting.
	[DEBUG] Found 2 non resonnant line families to fit
	[DEBUG] Performing fitting of family balmer
	[DEBUG] LSQ Fitting of 9 lines
	[DEBUG] added 9 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 85 iterations, redChi2 = 249.531
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Performing fitting of family forbidden
	[DEBUG] LSQ Fitting of 13 lines
	[DEBUG] added 13 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 148 iterations, redChi2 = 256.458
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Found 0 resonnant line families to fit
	
The first fit found a velocity offset of 280 km/s, which will result in a better
continuum fit.

.. _advanced:

Advanced usage
++++++++++++++

While the basic usage will be convenient for most application, it is sometimes useful
to use directly the ``Platefit`` python class. We give a few examples below.

.. code::

   from pyplatefit import Platefit
   pf = Platefit()
   
The platefit object has various associated methods.

.. code::

   res_cont = pf.fit_cont(sp, z, vdisp=80)
   pf.info_cont(res_cont)

::

  [INFO] Spectrum: test_data/udf10_00002.fits
  [INFO] Cont fit status: Continuum fit successful
  [INFO] Cont Init Z: 0.41892
  [INFO] Cont Fit Metallicity: 0.00400
  [INFO] Cont Fit E(B-V): 1.17
  [INFO] Cont Chi2: 0.05
  
.. code::

   import matplotlib.pyplot as plt
   fig,ax = plt.subplots(1,1)
   pf.plot_cont(ax, res_cont)
   
.. image:: images/adv_fig1.png  

The final continuum (in blue) and the first fitted value (in red) are displayed.

The line fitting can now be done on the continuum subtracted spectrum.

.. code:: 

   res_line = pf.fit_lines(res_cont['line_spec'], z)
   
::

	[DEBUG] Getting lines from get_emlines...
	[DEBUG] 21.3 % of the spectrum is used for fitting.
	[DEBUG] Found 2 non resonnant line families to fit
	[DEBUG] Performing fitting of family balmer
	[DEBUG] LSQ Fitting of 9 lines
	[DEBUG] added 9 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 85 iterations, redChi2 = 249.367
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Performing fitting of family forbidden
	[DEBUG] LSQ Fitting of 13 lines
	[DEBUG] added 13 gaussian to the fit
	[DEBUG] Leastsq fitting with ftol: 1e-06 xtol: 1e-04 maxfev: 1000
	[DEBUG] Fit succeeded. after 148 iterations, redChi2 = 255.486
	[DEBUG] Saving results to tablines and ztab
	[DEBUG] Found 0 resonnant line families to fit

A detailed fit report can be obtained as follows:
  
.. code::

    pf.info_lines(res_line, full_output=True)
    
::

	  FAMILY   VEL  VEL_ERR    Z     Z_ERR    Z_INIT VDISP VDISP_ERR SNRMAX SNRSUM SNRSUM_CLIPPED  NL NL_CLIPPED NFEV RCHI2
	--------- ----- ------- ------- -------- ------- ----- --------- ------ ------ -------------- --- ---------- ---- ------
	   balmer 82.14    3.86 0.41919 1.29e-05 0.41892 66.11      4.26  20.63  13.28          12.97   9          5   85 249.37
	forbidden 92.49    5.29 0.41923 1.76e-05 0.41892 66.65      6.07  13.94  16.84          20.69  13          6  148 255.49
  
More information can be given by reviewing directly the lmfit information for each family:

.. code::

	res_line['lmfit_forbidden'].params.pretty_print()
	
::

	Name                               Value      Min      Max   Stderr     Vary     Expr Brute_Step
	dv_forbidden                       92.49     -500      500    5.288     True     None     None
	forbidden_HEI3890_gauss_flux        1343        0      inf    392.8     True     None     None
	forbidden_HEI3890_gauss_l0          3890     -inf      inf        0    False     None     None
	forbidden_HEI5877_gauss_flux       907.9        0      inf    459.3     True     None     None
	forbidden_HEI5877_gauss_l0          5877     -inf      inf        0    False     None     None
	forbidden_NEIII3870_gauss_flux     401.9        0      inf    386.2     True     None     None
	forbidden_NEIII3870_gauss_l0        3870     -inf      inf        0    False     None     None
	forbidden_NEIII3967_gauss_flux     759.3        0      inf      386     True     None     None
	forbidden_NEIII3967_gauss_l0        3969     -inf      inf        0    False     None     None
	forbidden_NEV3427_gauss_flux     0.07583        0      inf    496.7     True     None     None
	forbidden_NEV3427_gauss_l0          3427     -inf      inf        0    False     None     None
	forbidden_NII6550_gauss_flux        4502        0      inf    691.9     True     None     None
	forbidden_NII6550_gauss_l0          6550     -inf      inf        0    False     None     None
	forbidden_NII6585_gauss_flux    1.16e+04        0      inf     1018     True     None     None
	forbidden_NII6585_gauss_l0          6585     -inf      inf        0    False     None     None
	forbidden_OI6302_gauss_flux        723.6        0      inf    953.4     True     None     None
	forbidden_OI6302_gauss_l0           6302     -inf      inf        0    False     None     None
	forbidden_OII3727_gauss_flux        4341        0      inf    426.1     True     None     None
	forbidden_OII3727_gauss_l0          3727     -inf      inf        0    False     None     None
	forbidden_OII3729_gauss_flux        6066        0      inf    435.2     True     None     None
	forbidden_OII3729_gauss_l0          3730     -inf      inf        0    False     None     None
	forbidden_OIII4364_gauss_flux      27.45        0      inf    346.3     True     None     None
	forbidden_OIII4364_gauss_l0         4364     -inf      inf        0    False     None     None
	forbidden_OIII4960_gauss_flux      654.2        0      inf    265.6     True     None     None
	forbidden_OIII4960_gauss_l0         4960     -inf      inf        0    False     None     None
	forbidden_OIII5008_gauss_flux       2216        0      inf    272.6     True     None     None
	forbidden_OIII5008_gauss_l0         5008     -inf      inf        0    False     None     None
	vdisp_forbidden                    66.65        5      300     6.07     True     None     None


    
The corresponding plot can be displayed with the following command:

.. code::

   fig,ax = plt.subplots(1,1)
   pf.plot_lines(ax, res_line)
   
.. image:: images/adv_fig2.png  

To compute the Equivalent Width one can use:

.. code::
   
   pf.comp_eqw(sp, res_cont['line_spec'], z, res_line['lines'])
   
the table ``lines`` is now completed with EQW and EQW_ERR columns.


.. _emlines:

Master table of emission lines
++++++++++++++++++++++++++++++

The master line information are taken from the MPDAF routine get_emlines
see `documentation <https://mpdaf.readthedocs.io/en/latest/api/mpdaf.sdetect.get_emlines.html#mpdaf.sdetect.get_emlines>`_. 
As shown later it is also possible to use its own line table. To review this master list use the following command:
   
.. code::

   from mpdaf.sdetect import get_emlines
   tab = get_emlines(table=True)
   tab.pprint_all()

::

       LINE    LBDA_OBS TYPE MAIN DOUBLET FAMILY VDISP  DNAME
    --------- -------- ---- ---- ------- ------ ----- -------
      LYALPHA  1215.67   em    1     0.0      3     0     Lyα
      NeV1238  1238.82   em    0  1240.8      2     0    None
      NeV1243   1242.8   em    0  1240.8      2     0     Nev
     SiII1260  1260.42   is    0     0.0      0     0    Siɪɪ
       OI1302  1302.17   is    0     0.0      0     0      Oɪ
     SIII1304  1304.37   is    0     0.0      0     0    Siɪɪ
      CII1334  1334.53   is    0     0.0      0     0     Cɪɪ
     SIIV1394  1393.76   is    0     0.0      0     0    None
     SIIV1403  1402.77   is    0     0.0      0     0    Siɪᴠ
      CIV1548   1548.2   em    1  1549.5      3     0    None
      CIV1551  1550.77   em    1  1549.5      3     0     Cɪᴠ
     FEII1608  1608.45   is    0     0.0      0     0    None
     FEII1611   1611.2   is    0     0.0      0     0    Feɪɪ
     HEII1640  1640.42   em    0     0.0      2     0    Heɪɪ
     OIII1660  1660.81   em    0     0.0      2     0    None
     OIII1666  1666.15   em    0     0.0      2     0   Oɪɪɪ]
     ALII1671  1670.79   is    0     0.0      0     0    Alɪɪ
       AL1854   1854.1   is    0     0.0      0     0    None
       AL1862  1862.17   is    0     0.0      0     0   Alɪɪɪ
     CIII1907  1906.68   em    1  1907.7      2     0    None
     CIII1909  1908.73   em    1  1907.7      2     0   Cɪɪɪ]
      CII2324  2324.21   em    0  2326.0      2     0    None
      CII2326  2326.11   em    0  2326.0      2     0    Cɪɪ]
      CII2328  2327.64   em    0  2326.0      2     0    None
      CII2329  2328.84   em    0  2326.0      2     0    None
     FEII2344  2344.21   is    0     0.0      0     0    None
     FEII2374  2374.46   is    0     0.0      0     0    None
     FEII2383  2382.76   is    0     0.0      0     0    Feɪɪ
     NEIV2422  2421.83   em    0  2423.0      2     0    None
     NEIV2424  2424.42   em    0  2423.0      2     0    Neɪᴠ
     FEII2587  2586.65   is    0     0.0      0     0    None
     FEII2600  2600.17   is    0     0.0      0     0    Feɪɪ
     MGII2796  2796.35   em    0  2800.0      3     0    None
     MGII2803  2803.53   em    0  2800.0      3     0    Mgɪɪ
      MGI2853  2852.97   is    0     0.0      0     0     Mgɪ
      NEV3427  3426.85   em    0     0.0      2     0     Neᴠ
      OII3727  3727.09   em    1  3727.5      2     0    None
      OII3729  3729.88   em    1  3727.5      2     0   [Oɪɪ]
          H11   3771.7   em    0     0.0      1     0     H11
          H10  3798.98   em    0     0.0      1     0     H10
           H9  3836.47   em    0     0.0      1     0      H9
    NEIII3870  3870.16   em    1     0.0      2     0 [Neɪɪɪ]
          CAK  3933.66   is    0     0.0      0     0    None
          CAH  3968.45   is    0     0.0      0     0    CaHK
      HEI3890  3889.73   em    0     0.0      2     0    None
           H8  3890.15   em    0     0.0      1     0      H8
    NEIII3967  3968.91   em    0     0.0      2     0    None
     HEPSILON   3971.2   em    0     0.0      1     0      Hε
       HDELTA  4102.89   em    1     0.0      1     0      Hδ
          CAG  4304.57   is    0     0.0      0     0   Gband
       HGAMMA  4341.68   em    1     0.0      1     0      Hγ
     OIII4364  4364.44   em    0     0.0      2     0    None
        HBETA  4862.68   em    1     0.0      1     0      Hβ
     OIII4960   4960.3   em    1     0.0      2     0    None
     OIII5008  5008.24   em    1     0.0      2     0  [Oɪɪɪ]
          MGB  5175.44   is    0     0.0      0     0     Mgb
      HEI5877  5877.25   em    0     0.0      2     0    None
          NAD  5891.94   is    0     0.0      0     0     NaD
       OI6302  6302.05   em    0     0.0      2     0    [Oɪ]
      NII6550  6549.85   em    0     0.0      2     0    None
       HALPHA  6564.61   em    1     0.0      1     0      Hα
      NII6585  6585.28   em    1     0.0      2     0    None
      SII6718  6718.29   em    1     0.0      2     0    None
      SII6733  6732.67   em    1     0.0      2     0   [Sɪɪ]
    ARIII7138   7137.8   em    0     0.0      2     0 [Arɪɪɪ]

The FAMILY column encode the line family:

   - 1 : Balmer lines
   - 2 : Forbidden lines
   - 3 : Resonant lines
   
The TYPE column encode the line type:

   - em : emission 
   - is : absorption

The MAIN column is a flag to select only main lines

The DOUBLET column is used to identify multiplet. If non 0, all lines with the same DOUBLET wavelength are identified as multiplet

The VDISP column is reserved for future use

The DNAME is used for display

It is possible to use its own line list table by providing an astropy table with the following columns:

    - LINE : Line identifier (string)
    - FAMILY: the line family (0-3: 0=abs, 1=Balmer, 2=Forbidden, 3=Resonant)
    - LBDA_REST: the rest frame wavelength in vacuum (Angstroem)
    - TYPE: the line type (em/is)
    - DOUBLET: the central wavelength if this is a multiplet (Angstroem)
    - VDISP: boolean to fit the velocity dispersion independantly [currently not used]
    - DNAME: the line display name (None if no display)
 
 The parameter ``lines=table`` can then be used in ``fit_spec`` 

.. _parameters:
   
Line fitting default parameters
+++++++++++++++++++++++++++++++

Most of the parameters can be changed using the ``linepars`` dictionary in 
`fit_spec` or the `Platefit` class initialisation.

Here is the complete list of parameters:

    - (vel_min, vel, vel_max) : initial value of velocity offset in km/s and bounds
    - (vdisp_min, vdisp, vdisp_max) : initial value of velocity dispersion in km/s and bounds
    - (vdisp_min_lya, vdisp_lya, vdisp_max_lya) : initial value of velocity dispersion for lyalpha line  in km/s and bounds
    - (gamma_min, gamma_lya, gamma_max) : initial value and bounds for the skeness parameter of the lyalpha line 
    - (gamma_2lya1_min, gamma_2lya1, gamma_2lya1_max) : initial value and bounds for the left lyalpha line skeness parameter (only for double lyman alpha fit)
    - (gamma_2lya2_min, _2lya2, gamma_2lya2_max) : same for the left lyalpha line
    - (sep_2lya_min, sep_2lya, sep_2lya_max) : initial value and bounds for the rest frame peak separation (km/s) of the two lyalpha lines (only for double lyalpha fit)
    - delta_vel : maximum excursion in km/s of velocity offset with respect to the LSQ result when the EMCEE fit is performed 
    - delta_vdisp : same for velocity dispersion
    - delta_gamma : same for the gamma skewness parameter
    - windmax : half size of the window to perform a preliminary search of the lyalpha peak (used when the option find_lya_velocity_offset is activated)
    - xtol : relative error in the solution for the LSQ fit
    - ftol : relative error in the sum of square for the LSQ fit
    - maxfev : maximum allowed of function evaluation (LSQ fit)
    - steps : number of steps (EMCEE routine)
    - nwalkers : number of walkers (EMCEE routine), if 0 it is computed as 3 times the number of variables
    - burn : number of sample to discard (EMCEE routine)
    - seed : random generation see
    - progress : display progress bar during EMCEE optimisation
    - line_ratios : list of line ratios (CIII, OII, MGII) see the section :ref:`doublet`.
