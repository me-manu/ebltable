ebltable
========

.. image:: https://img.shields.io/pypi/v/ebltable
    :target: https://pypi.org/project/ebltable/
    :alt: Latest Release

.. image:: https://github.com/me-manu/ebltable/actions/workflows/python_publish.yml/badge.svg
    :target: https://github.com/me-manu/ebltable/actions/workflows/python_publish.yml
    :alt: Build

.. image:: https://readthedocs.org/projects/ebltable/badge/?version=latest
    :target: https://ebltable.readthedocs.io/en/latest/
    :alt: Documentation

.. image:: https://img.shields.io/github/license/me-manu/ebltable
    :target: https://github.com/me-manu/ebltable
    :alt: License
    
.. image:: https://img.shields.io/github/issues/me-manu/ebltable
    :target: https://github.com/me-manu/ebltable/issues
    :alt: Issues

.. image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7312062-blue
    :target: https://doi.org/10.5281/zenodo.7312062
    :alt: DOI

Python packages to read in and interpolate tables for the photon density
of the Extragalactic Background Light (EBL) and the resulting opacity
for high energy gamma rays.

The full documentation is available at https://ebltable.readthedocs.io/en/latest/.

Prerequisites
-------------

Python 3.8 or higher and the following packages:

    - numpy >= 1.19
    - scipy >= 1.7
    - astropy >= 4.0

Installation
------------

You can use pip to install the package:: 

    pip install ebltable

Example scripts and notebooks are provided on the github page in the
example/ and notebooks/ folder, https://github.com/me-manu/ebltable

Adding a New EBL Model
----------------------

If you would like to have your EBL model included in ebltable, please
`open a GitHub issue <https://github.com/me-manu/ebltable/issues>`_ and include
the following information:

1. **State your request** — briefly mention that you would like your model added.

2. **Provide a paper reference** — include the full citation or a link to the
   publication describing the EBL model (e.g. arXiv ID or DOI).

3. **Upload the data files** — attach two ASCII tables to the issue:

   *EBL intensity table*
      An *n* × *m* matrix of EBL intensities νI\ :sub:`ν` [nW m\ :sup:`-2` sr\ :sup:`-1`]
      for *n* redshifts and *m* wavelengths (in µm).
      The table must include a column for *z* = 0.
      See `ebl_saldana21_comoving.txt
      <https://github.com/me-manu/ebltable/blob/main/ebltable/data/ebl_saldana21_comoving.txt>`_
      for a worked example of the expected layout.

   *Optical depth table*
      An *n* × *m* matrix of optical depth values τ for *n* redshifts and
      *m* gamma-ray energies (in TeV or GeV — please specify the unit).
      See `ebl_dominguez11.out
      <https://github.com/me-manu/ebltable/blob/main/ebltable/data/ebl_dominguez11.out>`_
      for a worked example of the expected layout.

License
-------
ebltable is distributed under the modified BSD License.
