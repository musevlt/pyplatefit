Changelog
=========
v0.7 (XX/XX/2021)
-----------------
New features
^^^^^^^^^^^^
- update default line ratio range of OII doublet to 0.3-1.5
- one can now specify his own LSF model
- updated documentationn

Breaking Changes
^^^^^^^^^^^^^^^^

Bug fixes
^^^^^^^^^
- fix a bug with the option flag major_lines not correctly propagated

v0.6 (25/01/2021)
-----------------
New features
^^^^^^^^^^^^
- replace fortran routines (nnls) by python code and numpy call
- set default method to least_square using Trust Region Reflective method with 1.e-3 tolerance on parameters
- save method, status in ztable
- allow to use any minimizing methods and its corresponding input parameters from scipy minimize
- add info option in plot_fit to display values from the lines table
- values for line multiplet (eg OII3727b = OII3726 + OII3729) are returned in addition to the individual lines.

Breaking Changes
^^^^^^^^^^^^^^^^
- remove bootstrap option which was not given robust results and is now better replace by least_square 
- name of emission lines in default table have been updated to reflect common usage
- updated emission lines 

Bug fixes
^^^^^^^^^
- it is now possible to fix values by setting min=init=max in linepars dict (eg vdisp=(50,50,50).


v0.5 (17/01/2021)
-----------------
New features
^^^^^^^^^^^^
- add option n_cpu to speed up bootstrap computation using parallel computing
- add option fitlines to perform abs fitting only (Leindert)
- complete the reference line table with some UV emission lines
- add absorption line fitting
- compute line fit quality estimate for individual lines 
- replace the emcee method with bootstrap
- replace the mpdaf linelist module by a specific one in pyplatefit

Breaking Changes
^^^^^^^^^^^^^^^^
- Option comp_bic has been supressed in fit_spec

Bug fixes
^^^^^^^^^
- correct a bug in redshift value and error reported in ztable
- correct a bug in redshift error estimation


v0.4 (24/03/2020)
-----------------
New features
^^^^^^^^^^^^
- add the fit of double lyman-alpha line
- allow the use of its own lines table
- allow to define separately the bounds and starting value for lya
- update documentation 
- add the name of the line with maximum SNR in ztable
- line_plot function 
- the bounds for EMCEE fit are reduced and set with respect to the LSQ first fit
- improve unit tests 


Bug fixes
^^^^^^^^^
- correct bugs when using AO spectra with masked Na wavelength range 
- correct a bug in BIC computation for lyalpha
- set correctly the initial value of velocity dispersion for the fit
- correct a bug in the estimation of LBDA_OBS for lines other than lyalpha
- filter emcee warnings


v0.3 (19/10/2019)
--------------------
New features
^^^^^^^^^^^^
- Complete reorganisation and reoptimisation of the code
- Add documentation and tutorial


v0.2 (19/04/2019)
-----------------

Full rewriting of line fitting


v0.1 (12/04/2019)
-----------------

First released version
