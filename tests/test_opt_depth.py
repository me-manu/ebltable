from __future__ import absolute_import, division, print_function
import numpy as np
from numpy.testing import assert_allclose
from ebltable.tau_from_model import OptDepth


class TestOptDepth:

    def test_models(self):

        # all implemented models
        models = ['franceschini',
                  'franceschini2017',
                  'saldana-lopez',
                  'saldana-lopez-upper',
                  'saldana-lopez-lower',
                  'kneiske',
                  'finke',
                  'dominguez',
                  'dominguez-upper',
                  'dominguez-lower',
                  'inoue',
                  'inoue-low-pop3',
                  'inoue-up-pop3',
                  'gilmore',
                  'gilmore-fixed']

        ETeV = np.logspace(-2., 1.5, 100)
        z = np.arange(0.05, 0.75, 0.05)
        spectrum = lambda ETeV, **params: ETeV**params['index']
        params = {'index': -2.}

        for m in models:
            tau = OptDepth.readmodel(model=m, kx=1, ky=1)

            # test the setters and getters
            tau.z = tau.z
            tau.logEGeV = 10.**tau.logEGeV
            tau.tau = tau.tau

            assert_allclose(tau.opt_depth(0., ETeV), np.zeros_like(ETeV), atol=1e-6)

            # test the optical depth functions
            tau.opt_depth(z, ETeV)
            tau.opt_depth_inverse(z, tau=1.)
            tau.opt_depth_Ebin(z=0.1, Ebin=np.logspace(-1., 1., 16), func=spectrum, params=params)

    def test_writing_reading(self):
        model = 'saldana-lopez'

        ETeV = np.logspace(-2., 1.5, 100)
        z = np.arange(0.05, 0.75, 0.05)
        tau = OptDepth.readmodel(model=model, kx=1, ky=1)

        tau_val = tau.opt_depth(z, ETeV)
        tau.writefits('test.fits', z=z, ETeV=ETeV)
        tau_new = OptDepth.readfits('test.fits')

        assert_allclose(tau_new.tau.T, tau_val, atol=0.)