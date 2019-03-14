import os
import sys
from setuptools import find_packages
from numpy.distutils.core import setup, Extension

# The Fortran extension for nnls_burst
nnls_sources = ['mnbrak.f90', 'nnls_burst.f90', 'dbrent.f90',
                'fit_cont_nnls.f90']
ext = Extension('pyPLATEFIT.nnls_burst',
                [os.path.join('pyNNLS', f) for f in nnls_sources])

# Dependencies. Limit version of ppxf for Python 2 users
install_requires = ['numpy', 'matplotlib', 'astropy', 'scipy',
                    'PyAstronomy', 'lmfit', 'mpdaf']

PY2 = sys.version_info[0] == 2
if PY2:
    install_requires.append('ppxf<6.7.8')
else:
    install_requires.append('ppxf')


setup(
    name='pyPLATEFIT',
    version='0.1',
    description='TODO',
    author='Madusha Gunawardhana',
    author_email='gunawardhana@strw.leidenuniv.nl',
    packages=find_packages(),
    package_data={
        'pyPLATEFIT': ['BC03/bc_models_subset_cb08_miles_v1_bursts.fit']
    },
    zip_safe=False,
    install_requires=install_requires,
    ext_modules=[ext],
)
