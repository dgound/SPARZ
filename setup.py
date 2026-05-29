#create setup.py file for python package names SPARZ
#setup.py file is used to install the package using pip install command

from setuptools import setup, find_packages

setup(
    name='SPARZ',
    version='0.5',
    description='Compress Single Molecule Localization Microscopy Data',
    long_description='Compression of Single Molecule Localization Microscopy Data using Sparse matrices and h265 video compression',
    author='Dimos Gkountaroulis, Antonios Lioutas, Lian Jiang, and Laura Breimann',
    author_email='dimos.gkountaroulis@bcm.edu',
    url='https://github.com/your_username/your_package',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    py_modules=['SPARZ', 'cli', 'gui'],
    install_requires=[
        'av',
        'dask',
        'dask-image',
        'ffmpeg-python',
        'jinja2',
        'numpy',
        'pandas',
        'PyQt6',
        'scikit-image',
        'scipy',
        'sparse',
        'statsmodels',
        'tifffile',
        'tqdm',
        'zstandard',
    ],
    entry_points={
        'console_scripts': [
            'sparz=cli:main_compress',
            'unsparz=cli:main_decompress',
        ],
        'gui_scripts': [
            'sparz-gui=gui:main',
        ],
    },
)
