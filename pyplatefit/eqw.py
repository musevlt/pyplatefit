from logging import getLogger
import numpy as np
from astropy.convolution import Box1DKernel, convolve
from astropy.stats import sigma_clipped_stats
from astropy.table import MaskedColumn
from astropy import units as u
from astropy import constants


C = constants.c.to(u.km / u.s).value

class EquivalentWidth:
    """
    This class implement Equivalength Width computation
    """
    def __init__(self):
        self.logger = getLogger(__name__)
        self.nfwhm = 2.0
        self.window = 50
        self.sigma_clip = 5
        return


    #def smooth_cont(self, spec, linefit, kernels=(100,30)):
        #""" perform continuum estimation 
        
        #Parameters
        #----------
        #spec : mpdaf.obj.Spectrum
           #raw spectrum
        #linefit: mpdaf.obj.Spectrum
           #fitted emission lines
        #kernels: tuple
           #(k1,k2), k1 is the kernel size of the median filter, k2 of the box filter
           #default: (100,30)
           
        #Return:
        #sm_cont : mpdaf.obj.Spectrum
           #smooth continuum spectrum
        #"""
        
        #spcont = spec - linefit 
        #sm_cont = spcont.median_filter(kernel_size=kernels[0])
        #kernel = Box1DKernel(kernels[1])
        #sm_cont.data = convolve(sm_cont.data, kernel)
        
        #return sm_cont
        
    def comp_eqw(self, spec, line_spec, z, lines_table):
        """
        compute equivalent widths, add computed values in lines table
        
        Parameters
        ----------
        spec : mpdaf.obj.Spectrum
           raw spectrum 
        line_spec : mpdaf.obj.Spectrum
           continuum subtracted spectrum 
        z : float
           redshift
        lines_table: astropy.table.Table
           lines table          
           
        """
        wave = spec.wave.coord(unit=u.angstrom, medium='vacuum')/(1+z)
        data = spec.data*(1+z)
        line_data = line_spec.data*(1+z)
        
        # mask all emission lines
        for line in lines_table:
            l0 = line['LBDA_REST'] 
            fwhm = 2.355*line['VDISP']*l0/C
            ksel = np.abs(l0-wave) < self.nfwhm*fwhm
            data.mask[ksel] = True
            line_data.mask[ksel] = True
        
        
        for name in ['EQW', 'EQW_ERR', 'CONT_OBS', 'CONT', 'CONT_ERR']:
            if name not in lines_table.colnames:
                lines_table.add_column(MaskedColumn(name=name, dtype=np.float, length=len(lines_table), mask=True))
                lines_table[name].format = '.2f'
        
        for line in lines_table:
            name = line['LINE']
            l0 = line['LBDA_REST']
            if name == 'LYALPHA':
                l1,l2 = (l0,l0+2*self.window)
            else:
                l1,l2 = (l0-self.window,l0+self.window)
            ksel = (wave > l1) & (wave < l2)
            if data[ksel].count() == 0:
                continue
            spmean,spmed,tmp = sigma_clipped_stats(data[ksel])
            tmp,tmp2,stddev = sigma_clipped_stats(line_data[ksel], sigma=self.sigma_clip)
            line['CONT_OBS'] = spmean/(1+z)
            line['CONT'] = spmean
            # compute continuum error     
            line['CONT_ERR'] = stddev # FIXME must be the error on the mean, not the std
            # stddev/np.sqrt(npix_window_of_the_line)
            # compute EQW in rest frame
            spmean_err = stddev/np.sqrt(np.sum(ksel))
            if spmean > 0:
                eqw = abs(line['FLUX'])/spmean
                eqw_err = line['FLUX_ERR']/spmean + line['FLUX']*spmean_err/spmean**2
                if line['FLUX'] > 0:
                    line['EQW'] = -eqw
                else:
                    line['EQW'] = eqw
                line['EQW_ERR'] = eqw_err
