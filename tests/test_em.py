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
    for f in ['udf10_00002.fits','udf10_00056.fits']:
        if not os.path.exists(os.path.join(tmpdir, f)):
            shutil.copy(os.path.join(DATADIR, f), tmpdir)

    # FIXME: keeping temp directory for now
    # yield tmpdir
    # shutil.rmtree(tmpdir)

    return tmpdir

    
    
def test_fit_lines(workdir):
    os.chdir(workdir)
    pf = Platefit(minpars=dict(method='leastsq', xtol=1.e-3))
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    res_cont = pf.fit_cont(sp, z, vdisp=80)
    spline = res_cont['line_spec']
    assert spline.shape == (3681,)
     
    res_line = pf.fit_lines(spline, z)
    assert_allclose(res_line['lmfit_balmer'].redchi,249.37,rtol=1.e-2)
    t = res_line['lines']
    r = t[t['LINE']=='OIII5007'][0]
    assert_allclose(r['VEL'],92.48,rtol=1.e-2)
    assert_allclose(r['Z'],0.41902,rtol=1.e-3)
    assert_allclose(r['LBDA_OBS'],7105.874,rtol=1.e-2)
    assert_allclose(r['FWHM_OBS'],4.381,rtol=1.e-2)
    assert_allclose(r['FLUX'],2215.83,rtol=1.e-3)
    assert_allclose(r['FLUX_ERR'],272.60,rtol=1.e-3)
    
    t = res_line['lines']
    r = t[t['LINE']=='OII3727b'][0]
    assert_allclose(r['VEL'],92.48,rtol=1.e-3)
    assert_allclose(r['Z'],0.41923,rtol=1.e-3)
    assert_allclose(r['LBDA_OBS'],5290.91,rtol=1.e-2)
    assert_allclose(r['FWHM_OBS'],3.78,rtol=1.e-2)
    assert_allclose(r['FLUX'],10406.67,rtol=1.e-3)
    assert_allclose(r['FLUX_ERR'],609.07,rtol=1.e-3)    
    
    ztab = res_line['ztable']
    assert 'balmer' in ztab['FAMILY']
    r = ztab[ztab['FAMILY']=='balmer'][0]
    assert_allclose(r['VEL'],82.14,rtol=1.e-3)
    assert_allclose(r['Z'],0.419309,rtol=1.e-4)
    assert r['NL'] == 9
    assert r['NL_CLIPPED'] == 5
    assert_allclose(r['SNRSUM_CLIPPED'],12.97,rtol=1.e-3)
    
def test_fit(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    
    pf = Platefit() 
    res = pf.fit(sp, z, lines=['HBETA'])
    
    r = res['lines'][0]
    assert r['LINE'] == 'HBETA'
    assert_allclose(r['VEL'],80.58,rtol=1.e-2)
    assert_allclose(r['VDISP'],64.45,rtol=1.e-2)
    assert_allclose(r['FLUX'],8477.96,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],98.40,rtol=1.e-2)
    assert_allclose(r['EQW'],-7.78,rtol=1.e-2)
    assert_allclose(r['EQW_ERR'],0.111,rtol=1.e-2)    
    
    
    r = res['ztable'][0]
    assert r['STATUS'] == 3
    assert_allclose(r['SNRSUM'],86.15,rtol=1.e-2)
    
def test_mpdaf(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    
    res = sp.fit_lines(z, lines=['HBETA'])
    
    r = res['lines'][0]
    assert r['LINE'] == 'HBETA'
    assert_allclose(r['VEL'],80.58,rtol=1.e-2)
    assert_allclose(r['VDISP'],64.45,rtol=1.e-2)
    assert_allclose(r['FLUX'],8477.96,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],98.40,rtol=1.e-2)
    assert_allclose(r['EQW'],-7.78,rtol=1.e-2)
    assert_allclose(r['EQW_ERR'],0.111,rtol=1.e-2)    
     
    r = res['ztable'][0]
    assert_allclose(r['SNRSUM'],86.15,rtol=1.e-2)
    
def test_fit_nocont(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    cont = sp.poly_spec(5)
    spline = sp - cont
    
    pf = Platefit(minpars=dict(method='leastsq', xtol=1.e-3)) 
    res = pf.fit(spline, z, lines=['HBETA'], fitcont=False)
    
    r = res['lines'][0]
    assert r['LINE'] == 'HBETA'
    assert_allclose(r['VEL'],85.27,rtol=1.e-2)
    assert_allclose(r['VDISP'],47.29,rtol=1.e-2)
    assert_allclose(r['FLUX'],6170.80,rtol=1.e-2)
    assert_allclose(r['FLUX_ERR'],347.62,rtol=1.e-2)
    
    r = res['ztable'][0]
    assert r['METHOD'] == 'leastsq'
    assert r['STATUS'] == 2
    
def test_fit_spec(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    
    res = fit_spec(sp, z)
    assert len(res['lines']) == 23   
    assert_allclose(res['dcont']['chi2'], 0.0471, rtol=1.e-2)
    assert_allclose(res['dline']['lmfit_balmer'].redchi, 249.36, rtol=1.e-2)
    
    res = fit_spec(sp, z, major_lines=True)
    assert len(res['lines']) == 11   
    
    res = fit_spec('udf10_00002.fits', z)
    assert_allclose(res['dcont']['chi2'], 0.0471, rtol=1.e-2)
    assert_allclose(res['dline']['lmfit_balmer'].redchi, 249.36, rtol=1.e-2)
    
    
    res = fit_spec(sp, z, lines=['OII3726','OII3729'], use_line_ratios=False)
    assert_allclose(res['dline']['lmfit_forbidden'].redchi, 11.47, rtol=1.e-2)
    
    res = fit_spec(sp, z, lines=['OII3726','OII3729'], ziter=True)
    assert_allclose(res['dline']['lmfit_forbidden'].redchi, 4.00, rtol=1.e-2)          
    
    res = fit_spec(sp, z, lines=['OII3726','OII3729'], use_line_ratios=True)
    lines = res['lines']
    lines = lines[~lines['ISBLEND']]
    ratio = lines['FLUX'][1]/lines['FLUX'][0]
    assert (ratio >= 0.3) and (ratio <= 1.5)
    
        
       
    res = fit_spec(sp, z, lines=['OII3726','OII3729'], use_line_ratios=True, 
                   linepars=dict(line_ratios=[("OII3726", "OII3729", 0.5, 0.8)]))
    lines = res['lines']
    lines = lines[~lines['ISBLEND']]
    ratio = lines['FLUX'][1]/lines['FLUX'][0]
    assert (ratio >= 0.5) and (ratio <= 0.8)    
    
    res = fit_spec(sp, z, lines=['OII3726','OII3729'], linepars=dict(vel=(0,0,0), vdisp=(20,20,20)))
    zfit = res['ztable'][0]['Z']
    assert_allclose(zfit, z, rtol=1.e-6)
    assert res['ztable'][0]['VEL'] == 0
    assert res['ztable'][0]['VDISP'] == 20
    
def test_fit_resonnant(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00056.fits')
    z = 1.30604
    res = fit_spec(sp, z, fit_all=True)
       
    res = fit_spec(sp, z)
    
    ztab = res['ztable']
    assert 'mgii2796' in ztab['FAMILY']
    r = ztab[ztab['FAMILY']=='mgii2796'][0]
    assert_allclose(r['VEL'],109.40,rtol=1.e-2)
    assert_allclose(r['VEL_ERR'],13.65,rtol=1.e-2)
    assert_allclose(r['VDISP'],50.59,rtol=1.e-2)
    assert_allclose(r['VDISP_ERR'],18.63,rtol=1.e-2)
    assert_allclose(r['SNRMAX'],5.27,rtol=1.e-2)
    assert_allclose(r['SNRSUM_CLIPPED'],5.27,rtol=1.e-2)
    assert_allclose(r['RCHI2'],11.79,rtol=1.e-2)
    assert r['NL'] == 1
    assert r['NL_CLIPPED'] == 1
    

def mylsf(lbda):
    return 2.00

def test_fit_lsf(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
  
    res = fit_spec(sp, z)
    lines = res['lines']
    r = lines[lines['LINE']=='OIII5007'][0]
    assert_allclose(r['VDISP'],66.65,rtol=1.e-2)
    assert_allclose(r['VDINST'],41.44,rtol=1.e-2)
    assert_allclose(r['FWHM_OBS'],4.38,rtol=1.e-2)
       
    res = fit_spec(sp, z, lsf=mylsf)
    lines = res['lines']
    r = lines[lines['LINE']=='OIII5007'][0]
    assert_allclose(r['VDISP'],72.69,rtol=1.e-2)
    assert_allclose(r['VDINST'],35.83,rtol=1.e-2)
    assert_allclose(r['FWHM_OBS'],4.52,rtol=1.e-2)    
    
    res = fit_spec(sp, z, lsf=None)
    lines = res['lines']
    r = lines[lines['LINE']=='OIII5007'][0]
    assert_allclose(r['VDISP'],84.04,rtol=1.e-2)
    assert_allclose(r['FWHM_OBS'],4.69,rtol=1.e-2)
    assert 'VDINST' not in lines.columns