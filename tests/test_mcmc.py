import os, shutil

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyplatefit import Platefit
from pyplatefit import fit_spec, get_lines
from astropy.table import Table
from mpdaf.obj import Spectrum

CURDIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(CURDIR, "test_data")
WORKDIR = ""


@pytest.fixture(scope="session")
def workdir(tmpdir_factory):
    """Ensure that build directory does not exists before each test."""
    tmpdir = str(tmpdir_factory.mktemp("pyplatefit_tests"))
    print("create tmpdir:", tmpdir)
    os.makedirs(tmpdir, exist_ok=True)
    for f in ['udf10_00053.fits','udf10_00723.fits','udf10_00106.fits']:
        if not os.path.exists(os.path.join(tmpdir, f)):
            shutil.copy(os.path.join(DATADIR, f), tmpdir)

    # FIXME: keeping temp directory for now
    # yield tmpdir
    # shutil.rmtree(tmpdir)

    return tmpdir

    
    
def test_faint_mcmc(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00723.fits')
    z = 3.18817
    
    res = fit_spec(sp, z, mcmc_lya=True, mcmcpars=dict(progress=False, steps=5000, save_proba=True))
    
    tab = res['ztable']
    r = tab[tab['FAMILY']=='lyalpha'][0]
    assert r['METHOD'] == 'emcee'
    assert (r['RCHAIN_CLIP'] > 0.7) and (r['RCHAIN_CLIP'] < 1.3)
    
    tab = res['lines']
    assert 'LYALPHA' in tab['LINE']
    r = tab[tab['LINE']=='LYALPHA'][0]
    assert_allclose(r['FLUX'],117.50,rtol=1.e-1)
    assert_allclose(r['FLUX_ERR'],28.5,rtol=1.e-1)
    assert (r['FLUX_RTAU'] > 1.0) and (r['FLUX_RTAU'] < 3.0)
    assert_allclose(r['Z_MIN99'],3.18574,rtol=1.e-4)
    

    

     
    

    
  
    
        
    

    
    
    

    
    
    
