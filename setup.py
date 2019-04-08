import os
from setuptools import find_packages
from numpy.distutils.core import setup, Extension

# The Fortran extension for nnls_burst
nnls_sources = ['mnbrak.f90', 'nnls_burst.f90', 'dbrent.f90',
                'fit_cont_nnls.f90']
ext = Extension('pyplatefit.nnls_burst',
                [os.path.join('pyNNLS', f) for f in nnls_sources])

# Dependencies. Limit version of ppxf for Python 2 users
install_requires = ['numpy', 'matplotlib', 'astropy', 'scipy',
                    'lmfit', 'mpdaf']

setup(
    name='pyplatefit',
    description='TODO',
    author='Madusha Gunawardhana',
    author_email='gunawardhana@strw.leidenuniv.nl',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    package_data={
        'pyplatefit': ['BC03/bc_models_subset_cb08_miles_v1_bursts.fit']
    },
    zip_safe=False,
    include_package_data=True,
    install_requires=install_requires,
    ext_modules=[ext],
)
