import numpy as np
import os
import sys

from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits
from logging import getLogger
from mpdaf.obj import airtovac, vactoair

from .cont_fitting import Contfit
from .line_fitting import Linefit


class Platefit:
    """
    This class is a poorman version of Platefit
    """

    def __init__(self):
        self.logger = getLogger(__name__)
        self.cont = Contfit()
        self.line = Linefit()

    def fit(self):
        """"""
                      
    def fit_cont(self, spec, z, vdisp):
        """"""
        return self.cont.fit(spec, z, vdisp)
    
    def fit_lines(self, line, z, return_lmfit_info=True):
        """"""
        return self.line.fit(line, z, return_lmfit_info=return_lmfit_info)        
        
    def info_cont(self, res):
        """
        print some info
        """
        self.cont.info(res)
        
    def info_lines(self, res):
        """
        print some info
        """
        self.line.info(res)    
           
            
    def plot(self, ax, res):
        """
        plot results
        """
        axc = ax[0] if type(ax) in [list, np.ndarray] else ax          
        if 'spec' in res:
            res['spec'].plot(ax=axc,  label='Spec')
        if 'cont_spec' in res:
            res['cont_spec'].plot(ax=axc, alpha=0.8, label='Cont Fit')
        axc.set_title('Cont Fit')
        axc.legend()
        if 'line_spec' in res:
            axc = ax[1] if type(ax) in [list, np.ndarray] else ax
            res['line_spec'].plot(ax=axc, label='Line')
            axc.set_title('Line')
            axc.legend()
        if 'line_fit' in res: 
            data_kws = dict(markersize=2)
            res['lmfit'].plot_fit(ax=axc, data_kws=data_kws, show_init=True)
            for row in res['table']:
                axc.axvline(row['LBDA_EXP'], color='r', alpha=0.2) 
                axc.axvline(row['LBDA'], color='k', alpha=0.2) 
                #y1,y2 = axc.get_ylim()
                #axc.text(row['LBDA']+5, y2-0.1*(y2-y1), row['LINE'], color='k', fontsize=8)                   
            axc.set_title('Line Fit')    
            
        
