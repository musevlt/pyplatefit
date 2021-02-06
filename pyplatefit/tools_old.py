"""Various utilities"""
import logging
import numpy as np
from astropy.table import Table, MaskedColumn
import shutil
from logging import getLogger
from joblib import delayed, Parallel

logger = getLogger(__name__)

def isnotebook():  # pragma: no cover
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def ProgressBar(*args, **kwargs):
    logger = logging.getLogger(__name__)
    if logging.getLevelName(logger.getEffectiveLevel()) == 'DEBUG':
        kwargs['disable'] = True

    from tqdm import tqdm, tqdm_notebook
    func = tqdm_notebook if isnotebook() else tqdm
    return func(*args, **kwargs)

def dict_values(d):
    """Return a list of all values in a dict."""
    return list(itertools.chain(*d.values()))

def iter_on_sources(srclist, fun, ncpu=1, outdir=None, rmdir=False, **kwargs):
    """ Iterate on sources
    srclist: list of source filenames
    fun: name of function which perform operation on one source
    ncpu: number of cpu
     outdir: name of output directory [None]
    rmdir: remove the output directory if it exist [False]"""
    if outdir is not None: 
        if rmdir:
            logger.warning('Delete %s directory', outdir)
            shutil.rmtree(outdir)    
        if not os.path.exists(outdir):
            logger.info('Creating output directory %s', outdir)
            os.mkdir(outdir)
        elif not os.path.isdir(outdir):
            logger.error('%s is not a directory', outdir)
            return 
        kwargs['outdir'] = oudir

    if ncpu > 1:    
        to_compute = []
        for name in srclist:
            to_compute.append(
                    delayed(fun)(name, **kwargs)
                )
        results = Parallel(n_jobs=ncpu)(ProgressBar(to_compute))
    else:
        for name in srclist:
            logger.debug('Performing operation for source %s', name)
            fun(name, **kwargs)     
    
    
