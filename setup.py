#create setup.py file for python package names SPARZ
#setup.py file is used to install the package using pip install command

from setuptools import setup

setup(
    name='SPARZ',
    version='0.5',
    description='Compress Single Molecule Localization Microscopy Data',
    long_description='Compression of Single Molecule Localization Microscopy Data using Sparse matrices and h265 video compression',
    author='Dimos Gkountaroulis, Antonios Lioutas, Lian Jiang, and Laura Breimann',
    author_email='dimos.gkountaroulis@bcm.edu',
    url='https://github.com/your_username/your_package',
    install_requires=[
        'numpy',
        'scikit-image',
        'scipy',
        'dask',
        'dask-image',
        'matplotlib',
        'scikit-video',
        'tifffile',
        'pims',
        'sparse',
        'av',
        'jinja2',
        'zstandard',
        'tqdm',
    ],
)