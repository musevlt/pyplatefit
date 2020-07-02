import os, shutil

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyplatefit import get_lines
from astropy.table import Table
from mpdaf.obj import Spectrum




    
def test_get_lines():

    tab = get_lines('LYALPHA')
    assert tab[0]['MAIN'] == True
    tab = get_lines(['LYALPHA','OII3727'])
    assert len(tab) == 2
    tab = get_lines('LYALPHA', z=3.0)
    assert_allclose(tab[0]['LBDA_OBS'], 4862.68, atol=1.e-2)
    tab = get_lines('LYALPHA', z=3.0, vac=False)
    assert_allclose(tab[0]['LBDA_OBS'], 4861.32, atol=1.e-2)
    
    tab = get_lines()
    assert len(tab) == 73
    tab = get_lines(main=True)
    assert len(tab) == 17
    tab = get_lines(main=False)
    assert len(tab) == 73 - 17
    tab = get_lines(doublet=True, family='forbidden')
    assert len(tab) == 22
    tab = get_lines(doublet=True, resonant=True)
    assert len(tab) == 8 
    tab = get_lines(lbrange=[4750,9350], z=0.5, resonant=False, family='balmer')
    assert len(tab) == 8  
    tab = get_lines(absline=True)
    assert len(tab) == 40  
    tab = get_lines(emiline=True, doublet=False)
    assert len(tab) == 29
    tab = get_lines(lbrange=[4750,9350], z=0.5, emiline=True)
    assert len(tab) == 18
    tab = get_lines(lbrange=[4750,9350], z=0.5, emiline=True, exlbrange=[5800,6000])
    assert len(tab) == 13
 
    
