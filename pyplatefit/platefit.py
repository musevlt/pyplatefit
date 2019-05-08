import logging
import datetime
from astropy.table import vstack, Column, MaskedColumn
from joblib import delayed, Parallel
from mpdaf.obj import Spectrum
from mpdaf.sdetect.source import Source

from .cont_fitting import Contfit
from .eqw import EquivalentWidth
from .line_fitting import Linefit, plotline
from .tools import ProgressBar

__all__ = ('Platefit', 'fit_all', 'fit_one', 'fit_spec')


class Platefit:
    """
    This class is a poorman version of Platefit.
    """

    def __init__(self, contpars={}, linepars={}):
        self.logger = logging.getLogger(__name__)
        self.cont = Contfit(**contpars)
        self.line = Linefit(**linepars)
        self.eqw = EquivalentWidth()

    def fit(self, spec, z, vdisp=80, major_lines=False, lines=None,
            emcee=False, use_line_ratios=True, vel_uniq_offset=False,
            lsf=True, eqw=True, trimm_spec=False):
        """Perform continuum and emission lines fit on a spectrum

        Parameters
        ----------
        line : mpdaf.obj.Spectrum
           continuum subtracted spectrum
        z : float
           reshift
        vdisp : float
           velocity dispersion in km/s [default 80 km/s]
        major_lines : bool
           if true, use only major lines as defined in MPDAF line list
           default False
        lines: list
           list of MPDAF lines to use in the fit
           default None
        emcee: bool
           if True perform a second fit using EMCEE to derive improved errors
           (note cpu intensive), default False
        use_line_ratios: bool
           if True, use constrain line ratios in fit
           default True
        vel_uniq_offset: bool
           if True, a unique velocity offset is fitted for all lines except
           lyman-alpha, default: False
        lsf: bool
           if True, use LSF model to take into account the instrumental LSF
           default: True
        eqw: bool
           if True compute equivalent widths

        Return
        ------
        result : dict
            - result['linetable']: astropy line table
            - result['ztable']; astropy z table
            - result['spec']: MPDAF original spectrum
            - result['cont']: MPDAF spectrum, fitted continuum in observed
              frame
            - result['line']: MPDAF spectrum, continnum removed spectrum in
              observed frame
            - result['linefit']: MPDAF spectrum, fitted emission lines in
              observed frame
            - result['fit']: MPDAF spectrum, fitted line+continuum in
              observed frame
            - result['res_cont']: return dictionary from fit_cont
            - result['res_line']: returned ResultObject from fit_lines

        """
        res_cont = self.fit_cont(spec, z, vdisp)
        res_line = self.fit_lines(res_cont['line_spec'], z,
                                  major_lines=major_lines, lines=lines,
                                  emcee=emcee, use_line_ratios=use_line_ratios,
                                  lsf=lsf, vel_uniq_offset=vel_uniq_offset, 
                                  trimm_spec=trimm_spec)

        if eqw:
            self.eqw.comp_eqw(spec, res_cont['line_spec'], z,
                              res_line.linetable)

        return dict(
            linetable=res_line.linetable,
            ztable=res_line.ztable,
            cont=res_cont['cont_spec'],
            line=res_cont['line_spec'],
            linefit=res_line.spec_fit,
            fit=res_line.spec_fit + res_cont['cont_spec'],
            spec=spec,
            res_cont=res_cont,
            res_line=res_line
        )

    def fit_cont(self, spec, z, vdisp):
        """Perform continuum lines fit on a spectrum

        Parameters
        ----------
        line : mpdaf.obj.Spectrum
            Continuum subtracted spectrum
        z : float
            Reshift
        vdisp : float
            Velocity dispersion in km/s

        Return
        ------
        result : dict
            - result['table_spec'] astropy table with the following columns:

              - RESTWL: restframe vacuum wavelength
              - FLUX: data value in restframe
              - ERR: stddev of data value
              - CONTFIT: continuum fit (restframe)
              - CONTRESID: smoothed continuum residual (restframe)
              - CONT: continuum fit + residual
              - LINE: continuum subtracted
              - AIRWL: observed wavelength in air

            - result['cont_spec']: MPDAF spectrum continuum (fit + smooth
              residual) in observed frame
            - result['cont_fit']: MPDAF spectrum continuum (fit only) in
              observed frame
            - result['line_spec']: MPDAF continuum subtracted spectrum in
              observed frame
            - result['success']: True or False
            - result['z']: Metallicity
            - result['ebv']: E(B-V)
            - result['chi2']: Chi2 fit
            - result['ages']: fitted ages
            - result['weights']: used weights

        """
        return self.cont.fit(spec, z, vdisp)

    
    def fit_lines(self, line, z, major_lines=False, lines=None, emcee=False, 
                  use_line_ratios=True, vel_uniq_offset=False, lsf=True, trimm_spec=False):
        """  
    Perform emission lines fit on a continuum subtracted spectrum 
    
    Parameters
    ----------
    line : mpdaf.obj.Spectrum
       continuum subtracted spectrum
    z : float
       reshift 
    major_lines : boolean
       if true, use only major lines as defined in MPDAF line list
       default False
    lines: list
       list of MPDAF lines to use in the fit
       default None
    emcee: boolean
       if True perform a second fit using EMCEE to derive improved errors (note cpu intensive)
       default False
    use_line_ratios: boolean
       if True, use constrain line ratios in fit
       default True
    vel_uniq_offset: boolean
       if True, a unique velocity offset is fitted for all lines except lyman-alpha
       default: False
    lsf: boolean
       if True, use LSF model to take into account the instrumental LSF
       default: True
    trimm_spec: boolean
       if True, mask wavelength regions outside +/- N*FWHM
        """
        return self.line.fit(line, z, major_lines=major_lines, lines=lines, emcee=emcee, 
                             use_line_ratios=use_line_ratios, 
                             lsf=lsf, vel_uniq_offset=vel_uniq_offset, trimm_spec=trimm_spec)        
        

    def info_cont(self, res):
        """ print some info """
        self.cont.info(res)

    def info_lines(self, res, full_output=False):
        """ print some info """
        self.line.info(res, full_output=full_output)

    def info(self, res, full_output=False):
        self.logger.info('++++ Continuum fit info')
        self.cont.info(res['res_cont'])
        self.logger.info('++++ Line fit info')
        self.line.info(res['res_line'], full_output=full_output)

    def eqw(self, lines_table, spec, smooth_cont, window=50):
        self.eqw.compute_eqw(lines_table, spec, smooth_cont, window=window)

    def plot_cont(self, ax, res):
        self.cont.plot(ax, res)

    def plot_lines(self, ax, res, start=False, iden=True, minsnr=0, line=None,
                   margin=5, dplot={'dl': 2.0, 'y': 0.95, 'size': 10}):
        self.line.plot(ax, res, start=start, iden=iden, minsnr=minsnr,
                       line=line, margin=margin, dplot=dplot)

    def plot(self, ax, res, mode=2, start=False, iden=True, minsnr=0,
             line=None, margin=5, dplot={'dl': 2.0, 'y': 0.95, 'size': 10}):
        """Plot fit results.

        Parameters
        ----------
        ax : axes
           matplotlib ax
        res: dictionary
           result as given by fit
        mode: integer
           plot mode: 0=continuum, 1=line, 2=cont+line
           default: 2
        start: bool
           use for plot mode = 1, display start fitting value
           default False
        iden: bool
           if True, display line names (mode=1,2 only)
        minsnr: float
           minimum SNR to display the line name (mode=1,2 only)
           default: 0
        line: str
           name of the line where to center the plot (eg HALPHA)
           defaut None, display the full spectrum
        margin: integer
           number of margin pixel to add for the zoom window
           default: 5
        dplot: dictionary
           dl: offset in A to display the line name [default 2]
           y: location of the line name in y (relative) [default 0.95]
           size: font size for label display [default 10]

        """
        if mode == 0:
            self.cont.plot(ax, res['res_cont'])
        elif mode == 1:
            self.line.plot(ax, res['res_line'], start=start, iden=iden,
                           minsnr=minsnr, line=line, margin=margin,
                           dplot=dplot)
        elif mode == 2:
            plotline(ax, res['spec'], res['fit'], res['cont'], None,
                     res['linetable'], iden=iden, minsnr=minsnr, line=line,
                     margin=margin, dplot=dplot)
        else:
            self.logger.error('unknown plot mode (0=cont,1=line,2=full)')


def fit_all(intable, colid='ID', colfrom='FROM', colz='Z', colspec='PATH', addcols=None,
            njobs=1, emcee=True, comp_bic=False, sourcetpl=None, prefix='PL'):
    """Fit all spectra from an input table.

    Parameters
    ----------
    intable: astropy.table.Table
       Input table
    colid: str
       Name of column with ID, default: ID
    colfrom: str
       Name of column with origin, default: FROM
    colz: str
       Name of column with input redshift, default: Z
    colspec: str
       Name of column with the spectrum object or path, default: PATH
    addcols: list of str
       Name of additional columns of the input table to add to the resulting tables
       default: None
    njobs: int
       Number of jobs to run in parallel, default: 1
    emcee: boolean
       if True estimate errors with MCMC
    comp_bic: boolean
       if True, estime BIC values from individual fit of lyman-alpha, oii and ciii
    source_tpl: str or None 
       source template format
       if not None, the sources are updated with all fitting results 
       example: /mydir/source_%05d.fits
    prefix: str
       prefix to add to table and spectra names added in the source 
       used only if source_tpl is not None
       default: PL

    Return
    ------
    ztable : astropy.table.Table
        Z information for each spectra
    ltable : astropy.table.Table
        line fit for each spectra

    """
    logger = logging.getLogger(__name__)
    
    if source_tpl is not None:
        # check if directory exist
        namedir = source_tpl.split('/')[:-1]
        if not os.path.isdir(namedir):
            raise OSError('Cannot find source directory %s'%(namedir))

    
    to_compute = [delayed(fit_one)(row[colid], row[colfrom], row[colz], row[colspec],
                                   row[addcols] if addcols is not None else None,
                                   emcee=emcee, comp_bic=comp_bic, 
                                   source_tpl=source_tpl, prefix=prefix)
                  for row in intable]
    results = Parallel(n_jobs=njobs)(ProgressBar(to_compute))
    ztab, ltab = zip(*results)
    ztable = vstack(ztab)
    ltable = vstack(ltab)
    return ztable, ltable


def fit_one(iden, detector, z, spec, addcols, source_tpl, prefix='PL', **kwargs):
    """ perform platefit cont and line fitting on a spectra
    """
    if isinstance(spec, str):
        spec = Spectrum(spec)

    logger = logging.getLogger(__name__)
    logger.debug('Fitting ID: %d FROM: %s Z: %.5f Path: %s',
                iden, detector, z, spec.filename)

    res = fit_spec(spec, z, **kwargs)
    ztab = res['ztable']
    ztab.add_column(Column(data=len(ztab) * [iden], name='ID'), index=0)
    ztab.add_column(Column(data=len(ztab) * [detector], name='FROM'), index=1)
    ztab['PATH'] = spec.filename
    ltab = res['linetable']
    ltab.add_column(Column(data=len(ltab) * [iden], name='ID'), index=0)
    ltab.add_column(Column(data=len(ltab) * [detector], name='FROM'), index=1)
    if addcols is not None:
        for key in addcols.colnames:
            ltab[key] = addcols[key]
            ztab[key] = addcols[key]
            
    if source_tpl is not None:
        # update the source
        srcname = source_tpl%iden
        src = Source.from_file(srcname)
        add_platefit_to_source(src, res, prefix=prefix)
        src.write(srcname)
        
    return ztab, ltab

def add_platefit_to_source(src, res, prefix='PL'):
    """ Add platefit info into an existing source
    src: MPDAF source
    res: result of platefit 
    prefix: to add to the tables and spectra names in source
    """
    ztable = res['ztable'].copy()
    ztable.remove_columns(['PATH','ID','FROM'])
    ltable = res['linetable'].copy()
    ltable.remove_columns(['ID','FROM'])
    
    ltable.remove_column('DNAME') # WIP this cannot be saved in fits ??
    
    now = datetime.datetime.now()
    src.tables[prefix+'_LINES'] = ltable
    src.tables[prefix+'_Z'] = ztable
    src.spectra[prefix+'_CONT'] = res['cont']
    src.spectra[prefix+'_LINE'] = res['line']
    src.spectra[prefix+'_LINEFIT'] = res['linefit']
    src.spectra[prefix+'_FIT'] = res['fit']
    src.add_attr(prefix+'_RUN', now.isoformat(), 'Date of Platefit Run')
    return


def fit_spec(spec, z, emcee=True, ziter=True, comp_bic=False):
    """ 
    perform platefit cont and line fitting on a spectra
    
    Parameters
    ----------
    spec: MPDAF spectrum
    z: redshift (in vacuum)
    emcee: use MCMC to estimate errors
    ziter: perform two successive fits
    comp_bic: compute BIC 
    
    """
    pl = Platefit()
    res1 = pl.fit(spec, z, emcee=emcee, vel_uniq_offset=True)
    ztab = res1['ztable']
    ltab = res1['linetable']
    if ziter:      
        ztab = ztab[ztab['FAMILY'] == 'all']      
        ltab = ltab[ltab['FAMILY'] == 'all']
        z2 = ztab[0]['Z']
        res2 = pl.fit(spec, z2, emcee=emcee, vel_uniq_offset=False)
        ztable2 = res2['ztable']
    if comp_bic:
        for name in ['BIC_LYALPHA','BIC_OII','BIC_CIII']:
            ztable2.add_column(MaskedColumn(name=name, dtype=float, length=len(ztable2), mask=True))
            ztable2[name].format = '.2f'
        # compute bic for individual lines (lya,oii,ciii)
        if 'OII3727' in res2['linetable']['LINE']:
            lines = ['OII3727','OII3729']
            res3 = pl.fit_lines(res2['line'], z2, emcee=False, lines=lines, trimm_spec=True)
            if 'forbidden' in ztable2['FAMILY']:
                ksel = ztable2['FAMILY'] == 'forbidden'
                ztable2['BIC_OII'][ksel] = res3.bic
            else:
                if 'all' in ztable2['FAMILY']:
                    ksel = ztable2['FAMILY'] == 'forbidden'
                    ztable2['BIC_OII'][ksel] = res3.bic
        if 'CIII1907' in res2['linetable']['LINE']:
            lines = ['CIII1907','CIII1909']
            res3 = pl.fit_lines(res2['line'], z2, emcee=False, lines=lines, trimm_spec=True)
            if 'forbidden' in ztable2['FAMILY']:
                ksel = ztable2['FAMILY'] == 'forbidden'
                ztable2['BIC_CIII'][ksel] = res3.bic
            else:
                if 'all' in ztable2['FAMILY']:
                    ksel = ztable2['FAMILY'] == 'forbidden'
                    ztable2['BIC_CIII'][ksel] = res3.bic            
        if 'LYALPHA' in res2['linetable']['LINE']:
            lines = ['LYALPHA']
            res3 = pl.fit_lines(res2['line'], z2, emcee=False, lines=lines, trimm_spec=True)
            if 'lyalpha' in ztable2['FAMILY']:
                ksel = ztable2['FAMILY'] == 'lyalpha'
                ztable2['BIC_LYALPHA'][ksel] = res3.bic
    
    if ziter:         
        ztab = vstack([ztab, ztable2])
        ltab = vstack([ltab, res2['linetable']])
    res2['ztable'] = ztab
    res2['linetable'] = ltab        
    
    return res2
