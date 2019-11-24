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

    
def test_fit_cont(workdir):
    os.chdir(workdir)
    pf = Platefit()
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    assert sp.shape == (3681,)
    
    res = pf.fit_cont(sp, z, vdisp=80)
    assert res['success']
    assert_allclose(res['chi2'],0.04710,rtol=1.e-4)
    assert len(res['table_spec']) == 5066 
    assert_allclose(res['table_spec']['FLUX'][2925],1149.8476,rtol=1.e-3)
    assert_allclose(res['table_spec']['CONTFIT'][2925],1168.7825219,rtol=1.e-3)
    assert_allclose(res['table_spec']['LINE'][2925],-0.925022444,rtol=1.e-3)
    
def test_fit_lines(workdir):
    os.chdir(workdir)
    pf = Platefit()
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    res_cont = pf.fit_cont(sp, z, vdisp=80)
    spline = res_cont['line_spec']
    assert spline.shape == (3681,)
     
    res_line = pf.fit_lines(spline, z, emcee=False)
    assert_allclose(res_line['lmfit_balmer'].redchi,249.37,rtol=1.e-4)
    t = res_line['lines']
    r = t[t['LINE']=='OIII5008'][0]
    assert_allclose(r['VEL'],92.48,rtol=1.e-3)
    assert_allclose(r['Z'],0.41923,rtol=1.e-3)
    assert_allclose(r['LBDA_OBS'],7105.874,rtol=1.e-2)
    assert_allclose(r['FWHM_OBS'],4.381,rtol=1.e-2)
    assert_allclose(r['FLUX'],2215.83,rtol=1.e-3)
    assert_allclose(r['FLUX_ERR'],272.60,rtol=1.e-3)
    
    ztab = res_line['ztable']
    assert 'balmer' in ztab['FAMILY']
    r = ztab[ztab['FAMILY']=='balmer'][0]
    assert_allclose(r['VEL'],82.14,rtol=1.e-3)
    assert_allclose(r['Z'],0.419196,rtol=1.e-5)
    assert r['NL'] == 9
    assert r['NL_CLIPPED'] == 5
    assert_allclose(r['SNRSUM_CLIPPED'],12.97,rtol=1.e-3)
    
def test_fit(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    
    pf = Platefit() 
    res = pf.fit(sp, z, emcee=False, lines=['HBETA'])
    
    r = res['lines'][0]
    assert r['LINE'] == 'HBETA'
    assert_allclose(r['VEL'],80.58,rtol=1.e-3)
    assert_allclose(r['VDISP'],64.45,rtol=1.e-3)
    assert_allclose(r['FLUX'],8477.96,rtol=1.e-3)
    assert_allclose(r['FLUX_ERR'],99.10,rtol=1.e-3)
    assert_allclose(r['EQW'],-7.778115,rtol=1.e-3)
    assert_allclose(r['EQW_ERR'],0.11148,rtol=1.e-3)    
    
    
    r = res['ztable'][0]
    assert_allclose(r['SNRSUM'],85.55,rtol=1.e-3)
    
def test_mpdaf(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    
    res = sp.fit_lines(z, lines=['HBETA'])
    
    r = res['lines'][0]
    assert r['LINE'] == 'HBETA'
    assert_allclose(r['VEL'],80.58,rtol=1.e-3)
    assert_allclose(r['VDISP'],64.45,rtol=1.e-3)
    assert_allclose(r['FLUX'],8477.96,rtol=1.e-3)
    assert_allclose(r['FLUX_ERR'],99.10,rtol=1.e-3)
     
    r = res['ztable'][0]
    assert_allclose(r['SNRSUM'],85.55,rtol=1.e-3)
    
def test_fit_nocont(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    cont = sp.poly_spec(5)
    spline = sp - cont
    
    pf = Platefit() 
    res = pf.fit(spline, z, emcee=False, lines=['HBETA'], fitcont=False)
    
    r = res['lines'][0]
    assert r['LINE'] == 'HBETA'
    assert_allclose(r['VEL'],85.27,rtol=1.e-3)
    assert_allclose(r['VDISP'],47.29,rtol=1.e-3)
    assert_allclose(r['FLUX'],6170.80,rtol=1.e-3)
    assert_allclose(r['FLUX_ERR'],347.62,rtol=1.e-3)
    
def test_fit_spec(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    
    res = fit_spec(sp, z)
    assert_allclose(res['dcont']['chi2'], 0.0471, rtol=1.e-3)
    assert_allclose(res['dline']['lmfit_balmer'].redchi, 249.37, rtol=1.e-3)
    
    res = fit_spec('udf10_00002.fits', z)
    assert_allclose(res['dcont']['chi2'], 0.0471, rtol=1.e-3)
    assert_allclose(res['dline']['lmfit_balmer'].redchi, 249.37, rtol=1.e-3)
    
    res = fit_spec(sp, z, lines=['OII3727','OII3729'], use_line_ratios=False)
    assert_allclose(res['dline']['lmfit_forbidden'].redchi, 11.47, rtol=1.e-3)
    
    res = fit_spec(sp, z, lines=['OII3727','OII3729'], use_line_ratios=True)
    assert_allclose(res['dline']['lmfit_forbidden'].redchi, 11.47, rtol=1.e-3)    
    
    
    res = fit_spec(sp, z, lines=['OII3727','OII3729'], use_line_ratios=True, 
                   linepars=dict(line_ratios=[("OII3727", "OII3729", 0.5, 0.8)]))
    assert_allclose(res['dline']['lmfit_forbidden'].redchi, 91.36, rtol=1.e-3)
     
     
def test_fit_resonnant(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00056.fits')
    z = 1.30604
    res = fit_spec(sp, z, fit_all=True)
    
    r = res['ztable'][0]
    assert r['FAMILY'] == 'all'
    assert_allclose(r['VEL'],80.68,rtol=1.e-2)
    assert_allclose(r['VEL_ERR'],0.897,rtol=1.e-2)
    assert_allclose(r['VDISP'],41.97,rtol=1.e-2)
    assert_allclose(r['VDISP_ERR'],1.080,rtol=1.e-2)
    assert_allclose(r['SNRMAX'],65.53,rtol=1.e-2)
    assert_allclose(r['SNRSUM_CLIPPED'],42.20,rtol=1.e-2)
    assert_allclose(r['RCHI2'],0.889,rtol=1.e-2)
    assert r['NL'] == 19
    assert r['NL_CLIPPED'] == 10
    
    res = fit_spec(sp, z)
    
    ztab = res['ztable']
    assert 'mgii2796' in ztab['FAMILY']
    r = ztab[ztab['FAMILY']=='mgii2796'][0]
    assert_allclose(r['VEL'],109.40,rtol=1.e-2)
    assert_allclose(r['VEL_ERR'],14.19,rtol=1.e-2)
    assert_allclose(r['VDISP'],50.59,rtol=1.e-2)
    assert_allclose(r['VDISP_ERR'],19.29,rtol=1.e-2)
    assert_allclose(r['SNRMAX'],5.38,rtol=1.e-2)
    assert_allclose(r['SNRSUM_CLIPPED'],5.38,rtol=1.e-2)
    assert_allclose(r['RCHI2'],12.47,rtol=1.e-2)
    assert r['NL'] == 2
    assert r['NL_CLIPPED'] == 1