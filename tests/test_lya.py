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
    for f in ['udf10_00053.fits','udf10_00723.fits','udf10_00106.fits']:
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
     
    res_line = pf.fit_lines(spline, z)
    assert_allclose(res_line['lmfit_lya'].redchi,2.201,rtol=1.e-2)
    tab = res_line['lines']
    r = tab[tab['LINE']=='LYALPHA'][0]
    assert r['LINE'] == 'LYALPHA'
    assert_allclose(r['VEL'],86.40,rtol=1.e-2)
    assert_allclose(r['Z'],4.77832,rtol=1.e-3)
    assert_allclose(r['FLUX'],4173.19,rtol=1.e-2)
    assert_allclose(r['SKEW'],7.25,rtol=1.e-2)
    assert_allclose(r['LBDA_OBS'],7022.60,rtol=1.e-2)
    assert_allclose(r['FWHM_OBS'],8.35,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],30.076,rtol=1.e-2)
    assert_allclose(r['SNR'],138.76,rtol=1.e-2)
   
    tab = res_line['ztable']
    r = tab[tab['FAMILY']=='lyalpha']
    assert_allclose(r['VEL'],86.39,rtol=1.e-2)
    assert_allclose(r['SNRSUM'],138.76,rtol=1.e-2)
    assert r['NL'] == 1
    
def test_fit(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00053.fits')
    z = 4.77666
    
    pf = Platefit(linepars=dict(seed=1, showprogress=False)) 
    res = pf.fit(sp, z, bootstrap=True, major_lines=True)
    
    r = res['lines'][0]
    assert r['LINE'] == 'LYALPHA'
    assert_allclose(r['VEL'],86.40,rtol=1.e-2)
    assert_allclose(r['Z'],4.77832,rtol=1.e-2)
    assert_allclose(r['FLUX'],4176.66,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],20.12,rtol=1.e-2)
    assert_allclose(r['SNR'],207.51,rtol=1.e-2)
    assert_allclose(r['EQW'],-60.65,rtol=1.e-2)
    assert_allclose(r['EQW_ERR'],1.70,rtol=1.e-2)
    assert_allclose(r['NSTD'],-2.295,rtol=1.e-2)
    
    
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
    assert_allclose(r['FLUX_ERR'],16.187,rtol=1.e-2)
    assert_allclose(r['SNR'],7.26,rtol=1.e-2)
    assert np.ma.is_masked(r['EQW'])
    
    assert 'HeII1640' in tab['LINE']
    r = tab[tab['LINE']=='HeII1640'][0]
    assert r['FLUX'] < 0.005
    #assert np.ma.is_masked(r['FLUX_ERR']) 
    #assert np.ma.is_masked(r['SNR'])   
    
    res = fit_spec(sp, z, lines=['LYALPHA','HeII1640'], bootstrap=True, linepars={'seed':1, 'showprogress':False})
    tab = res['lines']
    
    assert 'LYALPHA' in tab['LINE']
    r = tab[tab['LINE']=='LYALPHA'][0]
    assert_allclose(r['VEL'],160.45,rtol=1.e-2)
    assert_allclose(r['VDISP'],212.33,rtol=1.e-2)
    assert_allclose(r['FLUX'],121.01,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],28.54,rtol=1.e-2)
    assert_allclose(r['SNR'],4.24,rtol=1.e-2)
    assert_allclose(r['NSTD'],-1.31,rtol=1.e-2)
    assert np.ma.is_masked(r['EQW'])
    
    assert 'HeII1640' in tab['LINE']
    r = tab[tab['LINE']=='HeII1640'][0]
    assert_allclose(r['FLUX'],8.13,rtol=1.e-2)
    assert_allclose(r['SNR'],0.64,rtol=1.e-2)
    assert_allclose(r['NSTD'],-0.229,rtol=1.e-2)
    
    
    
def test_2lya(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00106.fits')
    z = 3.27554
    
    res = fit_spec(sp, z, lines=['LYALPHA'], dble_lyafit=True)
    
    tab = res['lines']
    assert 'LYALPHA1' in tab['LINE']
    r = tab[tab['LINE']=='LYALPHA1']
    assert_allclose(r['VEL'],34.15,rtol=1.e-2)
    assert_allclose(r['VDISP'],194.15,rtol=1.e-2)
    assert_allclose(r['FLUX'],680.54,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],21.03,rtol=1.e-2)
    assert_allclose(r['SNR'],32.35,rtol=1.e-2)
    assert_allclose(r['SEP'],515.76,rtol=1.e-2)
    assert_allclose(r['SEP_ERR'],9.66,rtol=1.e-2)
    assert 'LYALPHA2' in tab['LINE']
    r = tab[tab['LINE']=='LYALPHA2']
    assert_allclose(r['VEL'],34.15,rtol=1.e-2)
    assert_allclose(r['VDISP'],307.69,rtol=1.e-2)
    assert_allclose(r['FLUX'],1080.50,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],26.04,rtol=1.e-2)
    assert_allclose(r['SNR'],41.49,rtol=1.e-2)
    assert_allclose(r['SEP'],515.76,rtol=1.e-2)
    assert_allclose(r['SEP_ERR'],9.66,rtol=1.e-2) 
     
     
    

    
  
    
        
    

    
    
    

    
    
    
