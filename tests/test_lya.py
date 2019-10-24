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
    for f in ['udf10_00053.fits','udf10_00723.fits']:
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
    assert_allclose(res_line['lmfit_lyalpha'].redchi,2.976,rtol=1.e-2)
    assert res_line['lmfit_lyalpha'].nfev == 49
    r = res_line['lines'][0]
    assert r['LINE'] == 'LYALPHA'
    assert_allclose(r['VEL'],86.3987,rtol=1.e-2)
    assert_allclose(r['Z'],4.77691,rtol=1.e-5)
    assert_allclose(r['FLUX'],4172.96,rtol=1.e-2)
    assert_allclose(r['SKEW'],7.247,rtol=1.e-2)
    assert_allclose(r['LBDA_OBS'],7022.95,rtol=1.e-2)
    assert_allclose(r['FWHM_OBS'],8.351,rtol=1.e-2)
    assert_allclose(r['RCHI2'],2.976,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],34.96,rtol=1.e-2)
    assert_allclose(r['SKEW_ERR'],0.374,rtol=1.e-2)
    assert_allclose(r['SNR'],119.36,rtol=1.e-2)
   
    
    r = res_line['ztable'][0]
    assert_allclose(r['VEL'],86.40,rtol=1.e-2)
    assert_allclose(r['SNRSUM'],119.36,rtol=1.e-2)
    assert r['NL'] == 1
    
def test_fit(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00053.fits')
    z = 4.77666
    
    pf = Platefit(linepars=dict(seed=1)) 
    res = pf.fit(sp, z, emcee=True, major_lines=True)
    
    r = res['lines'][0]
    assert r['LINE'] == 'LYALPHA'
    assert_allclose(r['VEL'],86.3987,rtol=1.e-3)
    assert_allclose(r['Z'],4.77691,rtol=1.e-5)
    assert_allclose(r['FLUX'],4174.33,rtol=1.e-3)
    assert_allclose(r['FLUX_ERR'],20.346,rtol=1.e-1)
    assert_allclose(r['SNR'],210.50,rtol=1.e-2)
    assert_allclose(r['EQW'],-60.63,rtol=1.e-2)
    assert_allclose(r['EQW_ERR'],1.702,rtol=1.e-2)
    
    
def test_faint(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00723.fits')
    z = 3.18817
    
    res = fit_spec(sp, z)
    
    tab = res['lines']
    assert 'LYALPHA' in tab['LINE']
    r = tab[tab['LINE']=='LYALPHA'][0]
    assert_allclose(r['VEL'],37.03,rtol=1.e-2)
    assert_allclose(r['VDISP'],263.94,rtol=1.e-2)
    assert_allclose(r['FLUX'],117.54,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],16.48,rtol=1.e-2)
    assert_allclose(r['SNR'],7.13,rtol=1.e-2)
    assert np.ma.is_masked(r['EQW'])
    
    assert 'HEII1640' in tab['LINE']
    r = tab[tab['LINE']=='HEII1640'][0]
    assert_allclose(r['VEL'],-27.48,rtol=1.e-2)
    assert_allclose(r['VDISP'],120.2,rtol=1.e-2)
    assert_allclose(r['FLUX'],0.01,rtol=1.e-2)
    assert np.ma.is_masked(r['FLUX_ERR']) 
    assert np.ma.is_masked(r['SNR'])   
    assert np.ma.is_masked(r['EQW'])  
    
    res = fit_spec(sp, z, emcee=True, linepars={'seed':1})
    tab = res['lines']
    
    assert 'LYALPHA' in tab['LINE']
    r = tab[tab['LINE']=='LYALPHA'][0]
    assert_allclose(r['VEL'],37.03,rtol=1.e-2)
    assert_allclose(r['VDISP'],265.37,rtol=1.e-2)
    assert_allclose(r['FLUX'],113.66,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],21.28,rtol=1.e-2)
    assert_allclose(r['SNR'],5.34,rtol=1.e-2)
    assert np.ma.is_masked(r['EQW'])
    
    assert 'HEII1640' in tab['LINE']
    r = tab[tab['LINE']=='HEII1640'][0]
    assert_allclose(r['VEL'],-26.89 ,rtol=1.e-2)
    assert_allclose(r['VDISP'],128.92,rtol=1.e-2)
    assert_allclose(r['FLUX'],0.0106,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],0.0043,rtol=1.e-2)
    assert_allclose(r['SNR'],2.48,rtol=1.e-2) 
    assert np.ma.is_masked(r['EQW']) 
       
    

    
  
    
        
    

    
    
    

    
    
    
