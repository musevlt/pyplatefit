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
    for f in ['udf10_00002.fits']:
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
    assert_allclose(res_line.redchi,19.285,rtol=1.e-4)
    assert res_line.nfev == 277
    t = res_line.linetable
    r = t[t['LINE']=='OIII5008'][0]
    assert_allclose(r['VEL'],92.19,rtol=1.e-3)
    assert_allclose(r['Z'],0.41923,rtol=1.e-5)
    assert_allclose(r['FLUX'],2213.55,rtol=1.e-3)
    assert_allclose(r['FLUX_ERR'],74.82,rtol=1.e-3)
    
    r = res_line.ztable[0]
    assert r['FAMILY']=='balmer'
    assert_allclose(r['VEL'],82.67,rtol=1.e-3)
    assert_allclose(r['Z'],0.419196,rtol=1.e-5)
    assert r['NL'] == 9
    
def test_fit(workdir):
    os.chdir(workdir)
    
    sp = Spectrum('udf10_00002.fits')
    z = 0.41892
    
    pf = Platefit() 
    res = pf.fit(sp, z, emcee=False, lines=['HBETA'])
    
    r = res['linetable'][0]
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
    
    r = res['linetable'][0]
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
    
    r = res['linetable'][0]
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
    assert_allclose(res['res_cont']['chi2'], 0.027773, rtol=1.e-3)
    assert_allclose(res['res_line'].redchi, 14.342062863, rtol=1.e-3)
    
    res = fit_spec('udf10_00002.fits', z)
    assert_allclose(res['res_cont']['chi2'], 0.027773, rtol=1.e-3)
    assert_allclose(res['res_line'].redchi, 14.342062863, rtol=1.e-3)
    
    res = fit_spec(sp, z, ziter=False, lines=['OII3727','OII3729'], use_line_ratios=True)
    assert_allclose(res['res_line'].redchi, 11.46921247, rtol=1.e-3)
    
    
    res = fit_spec(sp, z, ziter=False, lines=['OII3727','OII3729'], use_line_ratios=True, 
                   linepars=dict(line_ratios=[("OII3727", "OII3729", 0.5, 0.8)]))
    assert_allclose(res['res_line'].redchi, 91.3640581, rtol=1.e-3)
     