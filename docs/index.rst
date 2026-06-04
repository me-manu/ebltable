Welcome to the ebltable documentation!
=======================================

ebltable is a Python package for reading, interpolating, and computing with
tables of the Extragalactic Background Light (EBL) photon density and the
resulting opacity for high-energy gamma rays.

The package provides two main classes:

* :py:class:`~ebltable.ebl_from_model.EBL` — interpolates EBL photon density
  models and computes the optical depth by numerical integration.
* :py:class:`~ebltable.tau_from_model.OptDepth` — interpolates pre-computed
  optical depth tables and provides helper functions for spectral analysis.

Getting Started
---------------

For installation instructions see the :ref:`installation` page.

For a quick start, take a look at the :ref:`tutorials`.

The full API is documented on the :ref:`module` page.

Getting Help
------------

If you encounter problems or have suggestions,
please open a `GitHub Issue <https://github.com/me-manu/ebltable/issues>`_.

Documentation Contents
======================

.. toctree::
   :caption: Getting Started
   :includehidden:
   :maxdepth: 2

   installation
   tutorials/index
   references

.. toctree::
   :caption: ebltable Package
   :includehidden:
   :maxdepth: 3

   module

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
