import os
from setuptools import find_packages
from numpy.distutils.core import setup, Extension

nnls_sources = ['mnbrak.f90', 'nnls_burst.f90', 'dbrent.f90',
                'fit_cont_nnls.f90']
ext = Extension('pyplatefit.nnls_burst',
                [os.path.join('pyNNLS', f) for f in nnls_sources])

setup(
    name='pyplatefit',
    version='0.1',
    description='TODO',
    author='Madusha Gunawardhana',
    author_email='gunawardhana@strw.leidenuniv.nl',
    packages=find_packages(),
    package_data={
        'pyplatefit': ['BC03/bc_models_subset_cb08_miles_v1_bursts.fit']
    },
    zip_safe=False,
    install_requires=['numpy', 'matplotlib', 'astropy', 'scipy', 'ppxf',
                      'PyAstronomy'],
    ext_modules=[ext],
)
