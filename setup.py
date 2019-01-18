import os
from setuptools import find_packages
from numpy.distutils.core import setup, Extension

nnls_sources = ['mnbrak.f90', 'nnls_burst.f90', 'dbrent.f90',
                'fit_cont_nnls.f90']
ext = Extension('pyPLATEFIT.nnls_burst',
                [os.path.join('pyNNLS', f) for f in nnls_sources])

setup(
    name='pyPLATEFIT',
    version='0.1',
    description='TODO',
    author='Madusha Gunawardhana',
    author_email='gunawardhana@strw.leidenuniv.nl',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy', 'matplotlib', 'astropy', 'scipy', 'ppxf',
                      'PyAstronomy'],
    ext_modules=[ext],
)
