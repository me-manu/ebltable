from setuptools import setup, find_packages 
from codecs import open
from os import path
from ebltable.version import __version__

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ebltable',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # http://packaging.python.org/en/latest/tutorial.html#version
    version=__version__,
    include_package_data = True,
    package_data={'ebltable': ['data/*'], },

    description='Python code to read in and interpolate tables for absorption of high energy gamma rays with additional helper functions',
    long_description=long_description,  #this is the

    # The project's main homepage.
    url='https://github.com/me-manu/ebltable',

    # Author details
    author='Manuel Meyer',
    author_email='me.manu@gmx.net',

    # Choose your license
    license='BSD',

    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords=['extragalactic backgroun light', 'Fermi', 'IACT', 'EBL', 'gamma-ray',
		'absorption', 'opacity', 'LIV', 'Lorentz invariance violation'],

    packages = find_packages(exclude=['build', 'docs', 'templates']),

    
    install_requires=[
        'numpy >= 1.6',
        'scipy',
        'astropy>=1.2.1',
    ]

)
