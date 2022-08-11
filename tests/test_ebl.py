from __future__ import absolute_import, division, print_function
import numpy as np
from numpy.testing import assert_allclose
from ebltable.ebl_from_model import EBL
from ebltable.tau_from_model import OptDepth
from astropy import units
from astropy import constants as c


class TestEBL:

    def test_models(self):

        # all implemented models
        models = ['franceschini',
                  'saldana-lopez',
                  'saldana-lopez-err',
                  'kneiske',
                  'finke',
                  'dominguez',
                  'dominguez-upper',
                  'dominguez-lower',
                  'cuba',
                  'gilmore',
                  'gilmore-fixed']

        z = np.arange(0., 4.2, 0.2)
        lmu = np.logspace(-1, 3, 4*16)

        for m in models:
            ebl = EBL.readmodel(model=m, kx=1, ky=1)

            # test the setters and getters
            ebl.z = ebl.z
            ebl.loglmu = 10.**ebl.loglmu
            ebl.nuInu = 10.**ebl.nuInu

            ebl.ebl_array(z, lmu)
            ebl.ebl_int(z=0.)
            ebl.n_array(z, c.c.value / lmu / 1e-6 * c.h.to('eV s').value)

    def test_writing_reading(self):
        model = 'saldana-lopez'

        lmu = np.logspace(-1., 3, 100)
        z = np.arange(0., 4.2, 0.2)
        ebl = EBL.readmodel(model=model, kx=1, ky=1)

        ebl_val = ebl.ebl_array(z, lmu)
        ebl.writefits('test.fits', z=z, lmu=lmu)
        ebl_new = EBL.readfits('test.fits')

        assert_allclose(10.**ebl_new.nuInu.T, ebl_val, rtol=1e-15)

    def test_optical_depth(self):
        model = 'saldana-lopez'
        ebl = EBL.readmodel(model=model, kx=1, ky=1)
        tau = OptDepth.readmodel(model=model, kx=1, ky=1)

        ETeV = np.logspace(-1.5, 1.5, 72)
        z_array = [0.1, 0.2, 0.5, 1., 2.]

        for z in z_array:
            tau_val = tau.opt_depth(z, ETeV)

            # calculate tau from EBL table using same cosmology
            # as in Saldana-Lopez et al. 2021
            tau_calc = ebl.optical_depth(z, ETeV, OmegaM=0.3, OmegaL=0.7, H0=70.)
            assert_allclose(tau_val, tau_calc, rtol=5.5e-2)
