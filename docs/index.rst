Welcome to pyplatefit documentation!
==========================================

pyplatefit is a python routine for emission and absorption lines fit of MUSE spectra. 
It is inspired by the IDL routine Platefit originally 
developed by Christy Tremonti, Jarle Brinchmann for the 
SDSS project (Tremonti et al. 2004;. Brinchmann et al. 2004).
The continuum fit is a direct python translation of the IDL routines. The emission and absorption line
fitting has been completely rewritten from scratch in python using the ``lmfit`` 
python fitting module.
Compared to the original IDL version, the python routine offer the following 
improvements:

  - Asymmetric gaussian single and double fit for the lyman-alpha line
  - Separate fit for resonant lines 
  - Allow various minimization algorithms
  - Improved Equivalent width estimate
  - Flexibility of a python code
  - Improved error estimate using MCMC ``emcee`` python module
  
The continuum fitting code has been developed by Madusha Gunawardhana
and the original line fitting code was developed by Yannick Roehlly. 
Jarle Brinchmann provide critical help in the validation process and explanation
of the original IDL code.
The final code is developed and maintained by Roland Bacon. If you use this software, please cite Bacon et al, 2023, A\&A, 670, A4.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorial
   changelog

API
---

.. automodapi:: pyplatefit 
   :no-inheritance-diagram:
   :inherited-members: Contfit


