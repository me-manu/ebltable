from __future__ import absolute_import, division, print_function
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ebltable.tau_from_model import OptDepth


@pytest.fixture(scope='module')
def tau():
    return OptDepth.readmodel(model='saldana-lopez', kx=1, ky=1)


class TestOptDepth:

    def test_models(self):
        models = OptDepth.get_models()

        ETeV = np.logspace(-2., 1.5, 100)
        z = np.arange(0.05, 0.75, 0.05)
        spectrum = lambda ETeV, **params: ETeV**params['index']
        params = {'index': -2.}

        for m in models:
            tau = OptDepth.readmodel(model=m, kx=1, ky=1)

            tau.x = 10.**tau.x
            tau.y = tau.y
            tau.Z = tau.Z

            assert_allclose(tau.opt_depth(0., ETeV), np.zeros_like(ETeV), atol=1e-6)

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

        assert_allclose(tau_new.Z.T, tau_val, atol=0.)

    # --- output shape tests ---

    def test_opt_depth_output_shape(self, tau):
        ETeV_arr = np.logspace(-1., 1., 20)
        z_arr = np.array([0.1, 0.5, 1.0])

        # scalar z, scalar E → scalar (0-d or squeezed)
        result = tau.opt_depth(0.5, 1.0)
        assert np.ndim(result) == 0 or result.shape == ()

        # scalar z, array E → 1-D array
        result = tau.opt_depth(0.5, ETeV_arr)
        assert result.shape == (len(ETeV_arr),)

        # array z, scalar E → 1-D array
        result = tau.opt_depth(z_arr, 1.0)
        assert result.shape == (len(z_arr),)

        # array z, array E → 2-D array (m × n)
        result = tau.opt_depth(z_arr, ETeV_arr)
        assert result.shape == (len(z_arr), len(ETeV_arr))

    # --- physical plausibility tests ---

    def test_opt_depth_zero_at_zero_redshift(self, tau):
        ETeV = np.logspace(-1., 1., 30)
        assert_allclose(tau.opt_depth(0., ETeV), np.zeros(len(ETeV)), atol=1e-6)

    def test_opt_depth_positive(self, tau):
        ETeV = np.logspace(-0.5, 1.5, 20)
        for z in [0.1, 0.5, 1.0, 2.0]:
            assert np.all(tau.opt_depth(z, ETeV) > 0.)

    def test_opt_depth_increases_with_redshift(self, tau):
        """At fixed energy, tau must increase with redshift."""
        z_vals = np.array([0.1, 0.3, 0.5, 1.0, 2.0])
        taus = tau.opt_depth(z_vals, 1.0)
        assert np.all(np.diff(taus) > 0.)

    def test_opt_depth_increases_with_energy(self, tau):
        """At fixed redshift, tau must increase with energy in the TeV range."""
        ETeV = np.logspace(0., 1.5, 10)  # 1–30 TeV, well above pair-production threshold
        taus = tau.opt_depth(1.0, ETeV)
        assert np.all(np.diff(taus) > 0.)

    # --- inverse consistency ---

    def test_opt_depth_inverse_consistency(self, tau):
        """opt_depth_inverse must return energy where tau equals the target value."""
        z = 0.5
        tau_target = 1.0
        E_GeV = tau.opt_depth_inverse(z, tau=tau_target)
        tau_check = tau.opt_depth(z, E_GeV / 1e3)  # GeV → TeV
        assert_allclose(float(tau_check), tau_target, rtol=0.05)

    def test_opt_depth_inverse_increases_with_redshift(self, tau):
        """At fixed tau, the energy horizon decreases as redshift increases
        (universe is more opaque at higher z, so same tau is reached at lower E)."""
        z_vals = np.array([0.1, 0.3, 0.5, 1.0])
        E_vals = tau.opt_depth_inverse(z_vals, tau=1.)
        assert np.all(np.diff(E_vals) < 0.)

    # --- opt_depth_Ebin ---

    def test_opt_depth_Ebin_shape(self, tau):
        n_bins = 10
        Ebin = np.logspace(-1., 1., n_bins + 1)
        spectrum = lambda E, **p: E ** p['index']
        result = tau.opt_depth_Ebin(z=0.3, Ebin=Ebin, func=spectrum, params={'index': -2.})
        assert result.shape == (n_bins,)

    def test_opt_depth_Ebin_positive(self, tau):
        Ebin = np.logspace(-0.5, 1.5, 9)
        spectrum = lambda E, **p: E ** p['index']
        result = tau.opt_depth_Ebin(z=1.0, Ebin=Ebin, func=spectrum, params={'index': -2.})
        assert np.all(result > 0.)
