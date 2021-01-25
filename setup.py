from setuptools import find_packages
from numpy.distutils.core import setup

# Dependencies.
install_requires = ['numpy', 'astropy', 'scipy', 'tqdm', 'joblib',
                    'lmfit', 'mpdaf', 'numdifftools']

setup(
    name='pyplatefit',
    description='Fit emission/absorption lines in MUSE spectra',
    author='Roland Bacon',
    author_email='roland.bacon@univ-lyon1.fr',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    package_data={
        'pyplatefit': ['BC03/bc_models_subset_cb08_milesx__bursts_extlam.fit',
                       'refdata/lines_table_platefit.fits']
    },
    zip_safe=False,
    include_package_data=True,
    install_requires=install_requires,
    ext_modules=[],
)
