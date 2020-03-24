Changelog
=========
v0.4dev (XX/XX/XX)
------------------
New features
^^^^^^^^^^^^
- replace the mpdaf linelist module by a specific one in pyplatefit
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
