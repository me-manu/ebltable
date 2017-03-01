ebltable
========

Python packages to read in and interpolate tables for the photon density of the Extragalactic Background Light (EBL)
and the resulting opacity for high energy gamma rays.

Prerequisites
-------------
Python 2.7 and newest versions of the following packages:
- numpy
- scipy
- astropy 

Installation
------------

Download the package, add the path of the repository to your python home variable,
e.g. by typing (or including in your .bashrc file):
> export PYTHONPATH="$PYTHONPATH:/path/to/ebltable:/some/other/path:"

To use EBL model files, you have to set the EBL_FILE_PATH environment variable, e.g., by typing 
> export EBL_FILE_PATH=/path/to/ebltable/ebl_model_files

The EBL model files are included in the ebl path. 
Check the installation by running 
> python example.py

The script is heavily commented to explain its use.

License
-------
eblstud is distributed under the modified BSD License.
