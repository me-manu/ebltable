.. _module:

API Reference
=============

This page documents the public API of ebltable.

EBL Photon Density
------------------

The :py:class:`~ebltable.ebl_from_model.EBL` class loads an EBL photon density
model and provides interpolation over redshift and wavelength. It can also
compute the optical depth by numerical integration over the line of sight.

A model is loaded with the class method :py:meth:`~ebltable.ebl_from_model.EBL.readmodel`:

.. code-block:: python

    from ebltable.ebl_from_model import EBL
    import numpy as np

    ebl = EBL.readmodel(model='saldana-lopez')

    # EBL intensity in nW/m^2/sr at z=0 for wavelengths 0.1–100 µm
    lmu = np.logspace(-1, 2, 100)
    nuInu = ebl.ebl_array(0., lmu)

    # Optical depth for 1 TeV photons at z=0.5
    tau = ebl.optical_depth(0.5, 1.0)

Available models can be listed with :py:meth:`~ebltable.ebl_from_model.EBL.get_models`.

.. autoclass:: ebltable.ebl_from_model.EBL
    :members:
    :undoc-members:
    :show-inheritance:

Optical Depth
-------------

The :py:class:`~ebltable.tau_from_model.OptDepth` class loads a pre-computed
optical depth table and provides fast interpolation over redshift and energy.

.. code-block:: python

    from ebltable.tau_from_model import OptDepth
    import numpy as np

    tau = OptDepth.readmodel(model='saldana-lopez')

    # Optical depth for 0.1–10 TeV photons at z=0.5
    ETeV = np.logspace(-1, 1, 50)
    tau_values = tau.opt_depth(0.5, ETeV)

.. autoclass:: ebltable.tau_from_model.OptDepth
    :members:
    :undoc-members:
    :show-inheritance:

Interpolation Base Class
------------------------

Both :py:class:`~ebltable.ebl_from_model.EBL` and
:py:class:`~ebltable.tau_from_model.OptDepth` inherit from the
:py:class:`~ebltable.interpolate.GridInterpolator` base class, which handles
the underlying 2-D B-spline interpolation.

.. autoclass:: ebltable.interpolate.GridInterpolator
    :members:
    :undoc-members:
    :show-inheritance:
