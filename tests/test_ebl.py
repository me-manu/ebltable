from __future__ import absolute_import, division, print_function
import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from astropy import constants as c
from ebltable.ebl_from_model import EBL
from ebltable.tau_from_model import OptDepth


@pytest.fixture(scope='module')
def ebl():
    return EBL.readmodel(model='saldana-lopez', kx=1, ky=1)


class TestEBL:

    def test_models(self):
        models = EBL.get_models()

        z = np.arange(0., 4.2, 0.2)
        lmu = np.logspace(-1, 3, 4*16)

        for m in models:
            ebl = EBL.readmodel(model=m, kx=1, ky=1)

            ebl.y = ebl.y
            ebl.x = 10.**ebl.x
            ebl.Z = 10.**ebl.Z

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

        assert_allclose(10.**ebl_new.Z.T, ebl_val, rtol=1e-15)

    def test_optical_depth(self):
        model = 'saldana-lopez'
        ebl = EBL.readmodel(model=model, kx=1, ky=1)
        tau = OptDepth.readmodel(model=model, kx=1, ky=1)

        ETeV = np.logspace(-1.5, 1.5, 72)
        z_array = [0.1, 0.2, 0.5, 1., 2.]

        for z in z_array:
            tau_val = tau.opt_depth(z, ETeV)
            tau_calc = ebl.optical_depth(z, ETeV, OmegaM=0.3, OmegaL=0.7, H0=70.)
            assert_allclose(tau_val, tau_calc, rtol=5.5e-2)

    # --- regression tests for issue #11 ---

    def test_optical_depth_scalar_energy(self, ebl):
        """Scalar float ETeV must not raise (was broken before issue #11 fix)."""
        result = ebl.optical_depth(0.5, 1.0)
        assert np.ndim(result) == 0
        assert float(result) > 0.

    def test_optical_depth_quantity_redshift(self, ebl):
        """astropy Quantity redshift must give same result as plain float."""
        ETeV = np.logspace(-1., 1., 20)
        result_qty = ebl.optical_depth(0.5 * u.one, ETeV)
        result_float = ebl.optical_depth(0.5, ETeV)
        assert_allclose(result_qty, result_float)

    # --- output shape tests ---

    def test_optical_depth_output_shape(self, ebl):
        # scalar energy → squeezed to 0-d scalar
        assert np.ndim(ebl.optical_depth(0.5, 1.0)) == 0

        # array energy → 1-D array of matching length
        ETeV = np.logspace(-1., 1., 30)
        assert ebl.optical_depth(0.5, ETeV).shape == (30,)

    # --- physical plausibility tests ---

    def test_optical_depth_positive(self, ebl):
        ETeV = np.logspace(-0.5, 1.5, 20)
        for z in [0.1, 0.5, 1.0, 2.0]:
            assert np.all(ebl.optical_depth(z, ETeV) > 0.)

    def test_optical_depth_increases_with_redshift(self, ebl):
        """At fixed energy, tau must increase with redshift."""
        taus = [float(ebl.optical_depth(z, 1.0)) for z in [0.1, 0.5, 1.0, 2.0]]
        assert all(taus[i] < taus[i + 1] for i in range(len(taus) - 1))

    def test_optical_depth_increases_with_energy(self, ebl):
        """At fixed redshift, tau must increase with energy in the TeV range."""
        ETeV = np.logspace(0., 1.5, 10)  # 1–30 TeV, well above threshold
        tau = ebl.optical_depth(1.0, ETeV)
        assert np.all(np.diff(tau) > 0.)

    # --- ebl_int ---

    def test_ebl_int_positive(self, ebl):
        for z in [0., 0.5, 1.0]:
            assert ebl.ebl_int(z) > 0.

    def test_ebl_int_convergence(self, ebl):
        """Result should converge as number of integration steps increases."""
        coarse = ebl.ebl_int(0., steps=50)
        fine = ebl.ebl_int(0., steps=500)
        assert_allclose(coarse, fine, rtol=0.01)

    # --- mean_free_path ---

    def test_mean_free_path_shape_and_sign(self, ebl):
        # meshgrid is (n_energies, n_z), so shape is (len(ETeV), len(z_array))
        z_array = np.linspace(0.1, 1.0, 5)
        ETeV = np.array([1.0, 10.0])
        result = ebl.mean_free_path(z_array, ETeV)
        assert result.shape == (len(ETeV), len(z_array))
        assert np.all(result > 0.)

    def test_mean_free_path_squeezed_for_scalar_inputs(self, ebl):
        """Single-element energy axis is squeezed, leaving only the z dimension."""
        z_array = np.linspace(0.1, 1.0, 5)
        result = ebl.mean_free_path(z_array, np.array([1.0]))
        assert result.ndim == 1
        assert result.shape == (len(z_array),)
