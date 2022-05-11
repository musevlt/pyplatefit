import os
from setuptools import find_packages
from numpy.distutils.core import setup, Extension

# The Fortran extension for nnls_burst
nnls_sources = ['mnbrak.f90', 'nnls_burst.f90', 'dbrent.f90',
                'fit_cont_nnls.f90']
ext = Extension('pyplatefit.nnls_burst',
                [os.path.join('pyNNLS', f) for f in nnls_sources])

install_requires = ['numpy', 'astropy', 'scipy', 'numdifftools',
                    'lmfit', 'mpdaf', 'more-itertools',
                    'emcee', 'pandas']

setup(
    name='pyplatefit',
    description='emission/absorption lines spectrum fitting',
    author='Roland Bacon',
    author_email='roland.bacon@univ-lyon1.fr',
    packages=find_packages(),
    package_data={
        'pyplatefit': ['BC03/*', 'refdata/lines_table_platefit.fits']
    },
    zip_safe=False,
    include_package_data=True,
    install_requires=install_requires,
    ext_modules=[ext],
)
