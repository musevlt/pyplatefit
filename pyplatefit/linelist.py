import numpy as np
import os
from astropy.table import Table
from matplotlib import transforms

from mpdaf.obj import airtovac, vactoair


__all__ = ['get_lines']

reftable = 'refdata/lines_table_platefit.fits'
CURDIR = os.path.dirname(os.path.abspath(__file__))

# list of useful emission lines
# name (id), vacuum wave A (c), line type (em/is) (tp), 
# main line(1/0) (s), doublet (average/0) (d)
# line family (0=abs, 1=Balmer, 2=Forbidden, 3=Resonant) (f)
# vdisp (0/1) (v), display name (n)



def get_lines(iden=None, orig=False,
              z=0, vac=True, 
              restframe=False,
              main=None, doublet=None,  
              family=None, resonant=None,
              absline=None, emiline=None,
              lbrange=None, exlbrange=None,
              user_linetable=None,
              margin=0):
    """Return a table of lines

    Parameters
    ----------
    iden : str or list of str
        identifiers, eg 'LYALPHA', ['OII3727','OII3729'] default None
    orig : bool
        if true return the original lines table 
    z : float
        redshift (None)
    vac : bool
        if False return wavelength in air    
    restframe : bool
        if true the wavelength are not reshifted but the
        selection with lbrange take into account the redshift
    main : bool or None
        if True select only major lines, if False only minor
    doublet : bool or None
        if true return only doublet, if false only singlet
    family : str or None
        select family (ism, balmer, forbidden)
    resonant : bool or None
        if True select resonant line, if False non resonant
    absline : bool or None
        if True select absorption lines, False non absorption lines
    emiline : bool or None
        if True select emission lines, False non emission lines
    lbrange : array-like
        wavelength range ex [4750,9350] default None
    exlbrange : array-like
        wavelength ranges to exclude in observed frame (ex for AO spectra), ex [5800,6000] or [[5000,5200],[6000,6500]]
    user_linetable : astropy Table
        user lines table, if None the default table is used
    margin : float
        margin in A to select a line (25)
    """
    
    if user_linetable is None:
        lines = get_line_table()
    else:
        lines = user_linetable.copy()
    if orig:
        return lines
    if iden is not None:
        if isinstance(iden, str):
            lines = lines[lines['LINE'] == iden]
        elif isinstance(iden, (list, tuple, np.ndarray)):
            lines = lines[np.in1d(lines['LINE'].tolist(), iden)]
    if not restframe:
        lines['LBDA_OBS'] = lines['LBDA_REST']*(1 + z)
        if not vac:
            lines['LBDA_OBS'] = vactoair(lines['LBDA_OBS'])
    if main is not None:
        if main:
            lines = lines[lines['MAIN']]
        else:
            lines = lines[~lines['MAIN']]
    if doublet is not None:
        if doublet:
            lines = lines[lines['DOUBLET']>0]
        else:
            lines = lines[lines['DOUBLET']<1.0]  
    if family is not None:
        lines = lines[lines['FAMILY']==family]
    if resonant is not None:
        if resonant:
            lines = lines[lines['RESONANT']]
        else:
            lines = lines[~lines['RESONANT']] 
    if absline is not None:
        if absline:
            lines = lines[lines['ABS']]
        else:
            lines = lines[~lines['ABS']] 
    if emiline is not None:
        if emiline:
            lines = lines[lines['EMI']]
        else:
            lines = lines[~lines['EMI']] 
    if lbrange is not None:
        if restframe:
            lbda = lines['LBDA_REST'] * (1 + z)
            lines = lines[(lbda - margin >= lbrange[0]) & (lbda - margin <= lbrange[1])]
        else:
            lines = lines[(lines['LBDA_OBS'] - margin >= lbrange[0]) & (lines['LBDA_OBS'] - margin <= lbrange[1])]        
    if exlbrange is not None:
        exlbs = exlbrange if any(isinstance(el, (list,np.ndarray)) for el in exlbrange) else [exlbrange]           
        for exlb in exlbs:
            if restframe:
                lines = lines[(lines['LBDA_REST']*(1+z) < exlb[0]) | (lines['LBDA_REST']*(1+z) > exlb[1])]
            else:
                lines = lines[(lines['LBDA_OBS']  < exlb[0]) | (lines['LBDA_OBS']  > exlb[1])]      
            
    return lines
    

def get_line_table():
    return Table.read(os.path.join(CURDIR, reftable))

def show_lines(ax, lines, dl=2, y=0.95, fontsize=10):
    trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes)              
    l1,l2,y1,y2 = ax.axis()
    for line in lines:
        l0 = line['LBDA_OBS']
        if (l0<l1) or (l0>l2):
            continue
        alpha = 0.8 if line['MAIN'] else 0.6
        color = 'r' if line['EMI'] else 'g'
        ax.axvline(l0, color=color, alpha=alpha)
        if line['DNAME'] != 'None':
            ax.text(line['LBDA_OBS']+dl, y, line['DNAME'], 
                    fontsize=fontsize, transform=trans)            
            
        