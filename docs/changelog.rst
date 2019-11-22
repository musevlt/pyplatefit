Changelog
=========
v0.4dev (XX/XX/XX)
------------------
New features
^^^^^^^^^^^^
- allow to define separately the bounds and starting value for lya
- add VDISP field in line table for future use (AGN type fitting, not yet implemented)
- update documentation 
- add the name of the line with maximum SNR in ztable
- line_plot function 
- the bounds for EMCEE fit are reduced and set with respect to the LSQ first fit
- improve unit tests 


Bug fixes
^^^^^^^^^
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
