from .version import __version__

from .line_fitting import fit_mpdaf_spectrum

def _setup_logging():
    import logging
    import sys
    from mpdaf.log import setup_logging
    setup_logging('mpdaf', level=logging.INFO, stream=sys.stdout)
    setup_logging(__name__, level=logging.DEBUG, stream=sys.stdout)

_setup_logging()



