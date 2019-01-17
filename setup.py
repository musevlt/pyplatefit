from setuptools import setup, find_packages

setup(
    name='pyPLATEFIT',
    version='0.1',
    description='TODO',
    author='Madusha Gunawardhana',
    author_email='gunawardhana@strw.leidenuniv.nl',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy', 'matplotlib', 'astropy', 'scipy',
                      'PyAstronomy'],
)
