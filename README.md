ebltable
========

Python packages to read in and interpolate tables for the photon density of the Extragalactic Background Light (EBL)
and the resulting opacity for high energy gamma rays.

Prerequisites
-------------
Python 2.7 and newest versions of the following packages:
- numpy
- scipy
- matplotlib

Installation
------------

Download the package, add the path of the repository to your python home variable,
e.g. by typing (or including in your .bashrc file):
> export PYTHONPATH="$PYTHONPATH:/path/to/repo:/some/other/path:"

To use EBL model files, you have to set the EBL_FILE_PATH environment variable, e.g., by typing 
> export EBL_FILE_PATH=/path/to/repro/ebl/ebl_model_files

The EBL model files are included in the ebl path. 
Check the installation by running 
> python example.py

The script is heavily commented to explain its use.

License
-------
eblstud is distributed under the modified BSD License.

- Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of the eblstud developers  nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE EBLSTUD DEVELOPERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
