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
    assert_allclose(res['z'],0.004,rtol=1.e-3)
    assert len(res['table_spec']) == 5066 
    assert_allclose(res['table_spec']['FLUX'][2925],1149.8476,rtol=1.e-3)
    assert_allclose(res['table_spec']['CONTFIT'][2925],1168.7825219,rtol=1.e-3)
    assert_allclose(res['table_spec']['LINE'][2925],-0.925022444,rtol=1.e-3)
    tab = res['table_spec']
    ksel = (tab['RESTWL']>3800) &  (tab['RESTWL']<4000)
    vals = [tab[key][ksel].mean() for key in ['FLUX','CONTFIT','CONTRESID','CONT','LINE']]
    assert_allclose(vals, [1037.10,1020.29,1.24,1021.53,15.57], rtol=1.e-2)  
    