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
    for f in ['udf10_00053.fits']:
        if not os.path.exists(os.path.join(tmpdir, f)):
            shutil.copy(os.path.join(DATADIR, f), tmpdir)

    # FIXME: keeping temp directory for now
    # yield tmpdir
    # shutil.rmtree(tmpdir)

    return tmpdir

    
def test_fit_cont(workdir):
    os.chdir(workdir)
    pf = Platefit()
    sp = Spectrum('udf10_00053.fits')
    z = 4.77666
    assert sp.shape == (3681,)
    
    res = pf.fit_cont(sp, z, vdisp=80)
    assert res['success']
    assert_allclose(res['chi2'],0.03302,rtol=1.e-4)
    assert len(res['table_spec']) == 5066 
    assert_allclose(res['table_spec']['FLUX'][2925],2771.5472399,rtol=1.e-3)
    assert_allclose(res['table_spec']['CONTFIT'][2925],49.5784686,rtol=1.e-3)
    assert_allclose(res['table_spec']['LINE'][2925],2734.1625427,rtol=1.e-3)
    
def test_fit_lines(workdir):
    os.chdir(workdir)
    pf = Platefit()
    sp = Spectrum('udf10_00053.fits')
    z = 4.77666
    res_cont = pf.fit_cont(sp, z, vdisp=80)
    spline = res_cont['line_spec']
    assert spline.shape == (3681,)
     
    res_line = pf.fit_lines(spline, z, emcee=False)
    assert_allclose(res_line.redchi,2.9597095,rtol=1.e-4)
    assert res_line.nfev == 108
    r = res_line.linetable[0]
    assert r['LINE'] == 'LYALPHA'
    assert_allclose(r['VEL'],75.00,rtol=1.e-3)
    assert_allclose(r['Z'],4.77691,rtol=1.e-5)
    assert_allclose(r['FLUX'],4172.96,rtol=1.e-3)
    assert_allclose(r['SKEW'],7.23,rtol=1.e-3)
    
    r = res_line.ztable[0]
    assert_allclose(r['VEL'],23.73,rtol=1.e-3)
    assert_allclose(r['Z'],4.77674,rtol=1.e-5)
    assert r['NL'] == 4
    
def test_fit(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00053.fits')
    z = 4.77666
    
    pf = Platefit(linepars=dict(seed=1)) 
    res = pf.fit(sp, z, emcee=True, major_lines=True)
    
    r = res['linetable'][0]
    assert r['LINE'] == 'LYALPHA'
    assert_allclose(r['VEL'],75.08,rtol=1.e-3)
    assert_allclose(r['Z'],4.77691,rtol=1.e-5)
    assert_allclose(r['FLUX'],4174.33,rtol=1.e-3)
    assert_allclose(r['FLUX_ERR'],23.59,rtol=1.e-3)
    
    
    r = res['ztable'][1]
    assert_allclose(r['SNRSUM'],176.92,rtol=1.e-3)
    

    
  
    
        
    

    
    
    

    
    
    
