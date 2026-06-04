from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ebltable',

    use_scm_version=True,
    include_package_data=True,
    package_data={'ebltable': ['data/*'], },

    description='Python code to read in and interpolate tables for absorption of high energy gamma rays with additional helper functions',
    long_description=long_description,  #this is the readme 
    long_description_content_type='text/x-rst',

    # The project's main homepage.
    url='https://github.com/me-manu/ebltable',

    # Author details
    author='Manuel Meyer',
    author_email='me.manu@gmx.net',

    # Choose your license
    license='BSD',

    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    python_requires='>=3.8',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],

    # What does your project relate to?
    keywords=['extragalactic backgroun light', 'Fermi', 'IACT', 'EBL', 'gamma-ray',
              'absorption', 'opacity', 'LIV', 'Lorentz invariance violation'],

    packages=find_packages(exclude=['build', 'docs', 'templates']),

    
    install_requires=[
        'numpy >= 1.19',
        'scipy >= 1.7',
        'astropy >= 4.0',
    ]

)
