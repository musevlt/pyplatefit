from .version import __version__
from .platefit import Platefit, fit_spec, plot_fit, print_res
from .cont_fitting import Contfit
from .line_fitting import Linefit, fit_lines, fit_abs
from .linelist import get_lines, show_lines        
from .eqw import EquivalentWidth


def _setup_logging():
    import logging
    import sys
    from mpdaf.log import setup_logging
    setup_logging(__name__, level=logging.DEBUG, stream=sys.stdout)


_setup_logging()
