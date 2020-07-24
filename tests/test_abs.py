import os, shutil

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyplatefit import Platefit
from pyplatefit import fit_spec
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
    for f in ['udf10_00002.fits','udf10_00056.fits','DR2_001028.fits']:
        if not os.path.exists(os.path.join(tmpdir, f)):
            shutil.copy(os.path.join(DATADIR, f), tmpdir)

    # FIXME: keeping temp directory for now
    # yield tmpdir
    # shutil.rmtree(tmpdir)

    return tmpdir

def test_fit_abs(workdir):
    os.chdir(workdir)
    sp = Spectrum('DR2_001028.fits')
    z = 1.90578
    res = fit_spec(sp, z, fitlines=False, fitabs=True)
    
  
    assert res['abs_fit'].shape == (3681,)
     
    t = res['lines']
    r = t[t['LINE']=='FeII2587'][0]
    assert_allclose(r['VEL'],-21.00,rtol=1.e-2)
    assert_allclose(r['NSTD'],-1.53,rtol=1.e-2)
    assert_allclose(r['SNR'],11.94,rtol=1.e-2)
    assert_allclose(r['FWHM_OBS'],6.76,rtol=1.e-2)
    assert_allclose(r['FLUX'],-203.42,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],17.10,rtol=1.e-2)
    assert_allclose(r['EQW'],2.53,rtol=1.e-2)
    
    ztab = res['ztable']
    assert 'abs' in ztab['FAMILY']
    r = ztab[ztab['FAMILY']=='abs'][0]
    assert_allclose(r['VEL'],-21.00,rtol=1.e-2)
    assert_allclose(r['Z'],1.90557,rtol=1.e-4)
    assert r['NL'] == 11
    assert r['NL_CLIPPED'] == 9
    assert_allclose(r['SNRSUM_CLIPPED'],27.68,rtol=1.e-2)
    assert_allclose(r['RCHI2'],0.79,rtol=1.e-2)
    
    res = fit_spec(sp, z, fitlines=False, fitabs=True, n_cpu=4, 
                   bootstrap=True, linepars=dict(seed=1, nbootstrap=40))
    
    t = res['lines']
    r = t[t['LINE']=='FeII2587'][0]
    assert_allclose(r['FLUX'],-208.53,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],19.17,rtol=1.e-2)
    assert_allclose(r['EQW'],2.59,rtol=1.e-2)
    
    ztab = res['ztable']
    assert 'abs' in ztab['FAMILY']
    r = ztab[ztab['FAMILY']=='abs'][0]
    assert_allclose(r['Z'],1.90557,rtol=1.e-4)
    assert r['NL'] == 11
    assert r['NL_CLIPPED'] == 9
    assert_allclose(r['SNRSUM_CLIPPED'],24.17,rtol=1.e-2)
    assert_allclose(r['RCHI2'],0.90,rtol=1.e-2)    

