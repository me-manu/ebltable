# ---- IMPORTS -----------------------------------------------#
from __future__ import absolute_import, division, print_function
import numpy as np
import os
import astropy.units as u
import astropy.constants as c
from collections.abc import Iterable
from scipy.integrate import simpson
from os.path import join
from astropy.cosmology import Planck15 as cosmo
from scipy.special import spence  # equals gsl_sf_dilog(1-z)
from .interpolate import GridInterpolator
# ------------------------------------------------------------#

# planck mass in eV
Mpl_eV = (np.sqrt(c.hbar * c.c / c.G) * c.c ** 2.).to('eV').value
# electron mass in eV
m_e_eV = (c.m_e * c.c ** 2.).to('eV').value
# Available models
models = ('franceschini',
          'kneiske',
          'dominguez',
          'dominguez-upper',
          'dominguez-lower',
          'saldana-lopez',
          'saldana-lopez-err',
          'gilmore',
          'gilmore-fixed',
          'finke',
          'finke2022',
          'cuba')

# ------------------------------------------------------------#
def p_kernel(x):
    """Kernel function from Biteau & Williams (2015), Eq. (7)"""

    x[x < 0.] = np.zeros(np.sum(x < 0.))
    x[x >= 1.] = np.zeros(np.sum(x >= 1.))
    x = np.sqrt(x)

    result = np.log(2.) * np.log(2.) - np.pi * np.pi / 6. \
        + 2. * spence(0.5 + 0.5 * x) - (x + x*x*x) / (1. - x*x) \
        + (np.log(1. + x) - 2. * np.log(2.)) * np.log(1. - x) \
        + 0.5 * (np.log(1. - x) * np.log(1. - x) - np.log(1. + x) * np.log(1. + x)) \
        + 0.5 * (1. + x*x*x*x) / (1. - x*x) * (np.log(1. + x) - np.log(1. - x))

    result[x <= 0.] = np.zeros(np.sum(x <= 0.))
    result[x >= 1.] = np.zeros(np.sum(x >= 1.))

    return result


class EBL(GridInterpolator):
    """
    Class to calculate EBL intensities from EBL models.
    """

    def __init__(self, z, lmu, nuInu, kx=1, ky=1, **kwargs):
        """
        Initiate EBL photon density model class. 

        Parameters
        ----------
        z: `~numpy.ndarray` or list
            source redshift, m-dimensional

        lmu: `~numpy.ndarray` or list
            Wavelengths in micro m

        nuInu: `~numpy.ndarray` or list
            n x m array with EBL photon density in nW / sr / m^2

        kx: int
            order of interpolation spline along energy axis, default: 2

        ky: int
            order of interpolation spline along energy axis, default: 2

        kwargs: dict
            Additional kwargs passed to `~scipy.interpolate.RectBivariateSpline`
        """
        self._model = kwargs.pop('model', None)
        super(EBL, self).__init__(lmu, z, nuInu, logx=True, logZ=True, kx=kx, ky=ky, **kwargs)
        
    @staticmethod
    def get_models():
        """Get the available EBL model strings and return them as a list"""
        return models

    @staticmethod
    def readmodel(model, kx=1, ky=1):
        """
        Read in an EBL model from an EBL model file

        Parameters
        ----------
        model: str,
            EBL model to use.
            Currently supported models are listed in Notes Section
        kx: int
            Spline order in x direction
        ky: int
            Spline order in y direction

        Notes
        -----
        Supported EBL models:
                Name:                Publication:
                franceschini        Franceschini et al. (2008)        http://www.astro.unipd.it/background/
                kneiske                Kneiske & Dole (2010)
                dominguez        Dominguez et al. (2011)
                dominguez-upper        Dominguez et al. (2011) upper uncertainty
                dominguez-lower        Dominguez et al. (2011) lower uncertainty
                saldana-lopez        Saldana-Lopez et al. (2021) https://www.ucm.es/blazars/ebl
                saldana-lopez-err    Saldana-Lopez et al. (2021) uncertainties,  https://www.ucm.es/blazars/ebl
                gilmore                Gilmore et al. (2012)                (fiducial model)
                gilmore-fixed   Gilmore et al. (2012)                (fixed model)
                finke                Finke et al. (2012)                (model C) http://www.phy.ohiou.edu/~finke/EBL/
                finke2022            Finke et al. (2022)                (model A) https://zenodo.org/record/7023073
                cuba                Haardt & Madua (2012)                http://www.ucolick.org/~pmadau/CUBA/HOME.html
        """
        ebl_file_path = os.path.join(os.path.split(__file__)[0],'data/')

        if model == 'kneiske':
            file_name = join(ebl_file_path, 'ebl_nuFnu_tanja.dat')
        elif model == 'franceschini':
            file_name = join(ebl_file_path, 'ebl_franceschini.dat')
        elif model == 'dominguez':
            file_name = join(ebl_file_path, 'ebl_dominguez11.out')
        elif model == 'dominguez-upper':
            file_name = join(ebl_file_path, 'ebl_upper_uncertainties_dominguez11.out')
        elif model == 'dominguez-lower':
            file_name = join(ebl_file_path, 'ebl_lower_uncertainties_dominguez11.out')
        elif model == 'gilmore':
            file_name = join(ebl_file_path, 'eblflux_fiducial.dat')
        elif model == 'gilmore-fixed':
            file_name = join(ebl_file_path, 'eblflux_fixed.dat')
        elif model == 'cuba':
            file_name = join(ebl_file_path, 'CUBA_UVB.dat')
        elif model == 'finke':
            file_name = join(ebl_file_path , 'ebl_modelC_Finke.txt')
        elif model == 'finke2022':
            file_name = os.path.join(ebl_file_path, 'EBL_nuInu_model_A_Finke2022.dat')
        elif model == 'saldana-lopez':
            file_name = join(ebl_file_path, 'ebl_saldana21_comoving.txt')
        elif model == 'saldana-lopez-err':
            file_name = join(ebl_file_path, 'eblerr_saldana21_comoving.txt')
        else:
            raise ValueError("Unknown EBL model chosen!")

        data = np.loadtxt(file_name)

        if model.find('gilmore') >= 0:
            z = data[0, 1:]
            lmu = data[1:, 0] * 1e-4  # convert from Angstrom to micro meter
            nuInu = data[1:, 1:]
            nuInu[nuInu == 0.] = 1e-20 * np.ones(np.sum(nuInu == 0.))
            
            # convert from ergs/s/cm^2/Ang/sr to nW/m^2/sr
            nuInu = (nuInu.T * data[1:, 0]).T * 1e4 * 1e-7 * 1e9

        elif model == 'cuba':
            z = data[0, 1:-1]
            lmu = data[1:, 0] * 1e-4
            nuInu = data[1:, 1:-1]

            # replace zeros by 1e-40
            idx = np.where(data[1:, 1:-1] == 0.)
            nuInu[idx] = np.ones(np.sum(nuInu == 0.)) * 1e-20

            # in erg / cm^2 / s / sr
            nuInu = (nuInu.T * c.c.value / (lmu * 1e-6)).T        
            nuInu *= 1e6        # in nW / m^2 /  sr

            # change to comoving units
            nuInu /= ((1. + z)**3.)[np.newaxis, :]

            # check where lmu is not strictly increasing
            idx = np.where(np.diff(lmu) == 0.)
            for i in idx[0]:
                lmu[i+1] = (lmu[i + 2] + lmu[i]) / 2.

        else:
            z = data[0, 1:]
            lmu = data[1:, 0]
            nuInu = data[1:, 1:]
            if model == 'finke': 
                lmu = lmu[::-1] * 1e-4
                nuInu = nuInu[::-1]

        return EBL(z, lmu, nuInu, model=model, kx=kx, ky=ky)

    @staticmethod
    def readascii(file_name, kx=1, ky=1, model_name=None, **kwargs):
        """
        Read in an EBL model file from an arbitrary file.

        Parameters
        ----------
        file_name: str
            full path to EBL photon density model file,
            with a (n+1) x (m+1) dimensional table.
            The zeroth column contains the wavelength values in mu meter,
            first row contains the redshift values.
            The remaining values are the EBL photon density values in nW / m^2 / sr.
            The [0,0] entry will be ignored.

        kx: int
            Spline order in x direction

        ky: int
            Spline order in y direction

        model_name: str or None,
            Name of EBL model

        kwargs: dict
            Additional kwargs passed to `~scipy.interpolate.RectBivariateSpline`
        """
        lmu, z, nuInu= GridInterpolator._read_ascii(file_name)
        return EBL(z, lmu, nuInu, model=model_name, kx=kx, ky=ky, **kwargs)

    @staticmethod
    def readfits(file_name,
                 hdu_nuInu_vs_z= 'NUINU_VS_Z',
                 hdu_wavelength='WAVELENGTHS',
                 zcol='REDSHIFT',
                 eblcol='EBL_DENS',
                 lcol='WAVELENGTH',
                 kx=1, ky=1,
                 model_name=None,
                 **kwargs):
        """
        Read EBL photon density from a fits file using the astropy.io module

        Parameters
        ----------
        filename: str, 
            full path to fits file containing the opacities, redshifts, and energies

        hdu_nuInu_vs_z: str
            optional, name of hdu that contains `~astropy.Table` with redshifts and tau values

        hdu_wavelengths: str
            optional, name of hdu that contains `~astropy.Table` with wavelegnths

        zcol: str
            optional, name of column of `~astropy.Table` with redshift values

        eblcol: str
            optional, name of column of `~astropy.Table` with EBL density values

        lcol: str
            optional, name of column of `~astropy.Table` with wavelength values

        kx: int
            Spline order in x direction

        ky: int
            Spline order in y direction

        model_name: str or None,
            Name of EBL model

        kwargs: dict
            Additional kwargs passed to `~scipy.interpolate.RectBivariateSpline`
        """

        lmu, z, nuInu = GridInterpolator._read_fits(file_name,
                                                    hdu_name_grid=hdu_nuInu_vs_z,
                                                    hdu_name_x=hdu_wavelength,
                                                    xcol_name=lcol,
                                                    ycol_name=zcol,
                                                    Zcol_name=eblcol,
                                                    xtarget_unit="um")

        return EBL(z, lmu, nuInu, model=model_name, kx=kx, ky=ky, **kwargs)

    def writefits(self, filename, z, lmu, overwrite=True):
        """
        Write optical depth to a fits file using 
        the astropy table environment. 

        Parameters
        ----------
        filename: str,
             full file path for output fits file

        z: array-like
            source redshift, m-dimensional

        lmu: array-like
            wavelenghts in micrometer, n-dimensional

        overwrite: bool
            Overwrite existing file.
        """
        self._write_fits(filename, lmu, z,
                         hdu_name_grid="NUINU_VS_Z",
                         hdu_name_x="WAVELENGTHS",
                         xunit="micrometer",
                         xcol_name="WAVELENGTH",
                         ycol_name="REDSHIFT",
                         Zcol_name="EBL_DENS",
                         xtarget_unit="micrometer", overwrite=overwrite)
        return

    def ebl_array(self, z, lmu):
        """
        Returns EBL intensity in nuInu [nW / m^2 / sr] 
        for redshift z and wavelength l (micron) from BSpline Interpolation

        Parameters
        ----------
        z: array-like
            source redshift, m-dimensional

        lmu: array-like
            wavelenghts in micrometer, n-dimensional

        Returns
        -------
        (m x n)-dim `~numpy.ndarray` with corresponding (nu I nu) values

        Notes
        -----
        if any z < self._z[0] (from interpolation table), 
        self._z[0] is used and RuntimeWarning is issued.

        """
        result = self.evaluate(lmu, z)
        return result

    def n_array(self, z, EeV):
        """
        Returns EBL photon density in [1 / cm^3 / eV] for redshift z and energy from BSpline Interpolation

        Parameters
        ----------
        z: array-like
            source redshift, n-dimensional

        EeV: array-like
            Energies in eV, m-dimensional

        Returns
        -------
        (N x M)-dim `~numpy.ndarray` with corresponding photon density values

        Notes
        -----
        if any z < self._z[0] (from interpolation table), self._z[0] is used and RuntimeWarning is issued.
        """
        if np.isscalar(EeV):
            EeV = np.array([EeV])
        elif EeV is Iterable:
            EeV = np.array(EeV)

        if np.isscalar(z):
            z = np.array([z])
        elif z is Iterable:
            z = np.array(z)

        # convert energy in eV to wavelength in micron
        # SI_h * SI_c / EeV / SI_e  * 1e6
        l = (c.h * c.c / (EeV * u.eV).to(u.J)).to(u.um).value
        # convert energy in J
        e_J = (EeV * u.eV).to(u.J).value

        n = self.ebl_array(z, l)
        # convert nuInu to photon density in 1 / J / m^3
        n = 4. * np.pi / c.c.value / e_J**2. * n * 1e-9
        # convert photon density in 1 / eV / cm^3 and return
        return n * c.e.value * 1e-6

    def ebl_int(self, z, lmin=0.01, lmax=1e3, steps=50):
        """
        Returns integrated EBL intensity in I [nW / m^2 / sr] 
        for redshift z between wavelegth lmin and lmax (micron) 

        Parameters
        ----------
        z: float
            redshift
        lmin: float
            minimum wavelength in micrometer
        lmax: float 
            maximum wavelength in micrometer
        steps: int
            number of steps for simps integration

        Returns
        -------
        Float with integrated nuInu value
        """
        logl = np.linspace(np.log10(lmin), np.log10(lmax), steps)
        lnl = np.log(np.linspace(lmin, lmax, steps))
        ln_Il = np.log(10.) * (self.ebl_array(z, 10.**logl))         # note: nuInu = lambda I lambda
        result = simps(ln_Il, lnl)
        return result

    def optical_depth(self, z0, ETeV,
                      OmegaM=0.3, OmegaL=0.7,
                      H0=70.,
                      steps_z=50,
                      steps_e=50,
                      egamma_LIV=True,
                      LIV_scale=0.,
                      nLIV=1):
        """
        calculates mean free path for gamma-ray energy ETeV at redshift z

        Parameters
        ----------
        z0: float
            source redshift

        ETeV: array-like
            Energies in TeV, n-dimensional

        H0: float,
            Hubble constant in km / Mpc / s. Default: 70.

        OmegaM: float,
            critical density of matter at z=0. Default: 0.3

        OmegaL: float,
            critical density of dark energy. Default: 0.7

        steps_z: int
            intergration steps for redshift (default : 50)

        steps_e: int
            number of integration steps for integration over EBL density (default: 60)

        LIV_scale: float
            Lorentz invariance violation parameter (quantum gravity scale),
            if 0. (default), do not include LIV

        nLIV: int
            order of LIV, only applied if LIV_scale > 0, default: 1

        egamma_LIV: bool
            if true, apply LIV to both electrons and photons

        Returns
        -------
        n-dim `~numpy.ndarray` with optical depth values

        Notes
        -----
        For calculation, see e.g.
        See Dwek & Krennrich 2013 or Mirizzi & Montanino 2009
        """
        if np.isscalar(ETeV):
            ETeV = np.array([ETeV])
        elif ETeV is Iterable:
            ETeV = np.array(ETeV)

        z_array = np.linspace(0., z0, steps_z)
        result = self.mean_free_path(z_array, ETeV,
                                     LIV_scale=LIV_scale,
                                     nLIV=nLIV,
                                     egamma_LIV=egamma_LIV,
                                     steps_e=steps_e)

        zz, ee = np.meshgrid(z_array, ETeV, indexing='ij')
        result = 1. / (result.T * u.Mpc).to(u.cm).value # this is in cm^-1
        # dt / dz for a flat universe
        result *= 1. / ((1. + zz) * np.sqrt((1. + zz)**3. * OmegaM + OmegaL) )

        result = simps(result, zz, axis=0)

        # convert from km / Mpc / s to 1 / s
        H0 = (H0 * cosmo.H0.unit).to('1 / s').value

        return result * c.c.to('cm / s').value /  H0

    def mean_free_path(self, z, ETeV,
                       steps_e=50,
                       LIV_scale=0.,
                       nLIV=1,
                       egamma_LIV=True):
        """
        Calculates mean free path in Mpc for gamma-ray energy ETeV at redshift z

        z: array-like
            redshift, n-dimensional
        ETeV: array-like
            Energies in TeV, m-dimensional
        steps_e: int
            number of integration steps for integration over EBL density (default: 60)
        LIV_scale: float
            Lorentz invariance violation parameter (quantum gravity scale),
            if 0. (default), do not include LIV
        nLIV: int
            order of LIV, only applied if LIV_scale > 0, default: 1
        egamma_LIV: bool
            if true, apply LIV to both electrons and photons

        Returns
        -------
        mxn-dim `~numpy.ndarray` with mean free path values in Mpc
        if m or n == 1, the axis will be squeezed, i.e. dropped.

        Notes
        -----
        For calculation, see e.g.
        see Dwek & Krennrich 2013 or Mirizzi & Montanino 2009.  For kernel see Biteau & Williams 2015
        """
        if np.isscalar(ETeV):
            ETeV = np.array([ETeV])

        elif ETeV is Iterable:
            ETeV = np.array(ETeV)

        if np.isscalar(z):
            z = np.array([z])
        elif z is Iterable:
            z = np.array(z)

        # max energy of EBL template in eV
        emax_eV = (c.h * c.c / (10.**np.min(self.x) * u.um)).to('eV').value

        # defines the effective energy scale for the LIV modification
        if LIV_scale:
            ELIV_eV = 4. * m_e_eV * m_e_eV * (LIV_scale * Mpl_eV)**nLIV
            if egamma_LIV:
                ELIV_eV /= 1. - 2.**(-nLIV)
            ELIV_eV = ELIV_eV ** (1./(2.+nLIV))

        # make z, ETeV to 2d arrays
        zz, EE_TeV = np.meshgrid(z, ETeV)  # m x n dimensional
        EEzz_eV = EE_TeV * 1e12 * (1. + zz)
        ethr_eV = m_e_eV * m_e_eV / EEzz_eV

        if LIV_scale:
            ethr_eV *= 1. + (EEzz_eV / ELIV_eV)**(nLIV + 2.)

        b3d_array = np.ones(ethr_eV.shape + (steps_e,)) * 1e-40
        e3d_array = np.ones(ethr_eV.shape + (steps_e,)) * 1e-40
        n3d_array = np.ones(ethr_eV.shape + (steps_e,)) * 1e-40

        for i in range(ethr_eV.shape[0]):  # loop over ETeV dimension
            for j in range(ethr_eV.shape[1]):  # loop over z dimension
                if ethr_eV[i,j] < emax_eV:
                    e3d_array[i,j] = np.logspace(np.log10(ethr_eV[i,j]), np.log10(emax_eV), steps_e)
                    b3d_array[i,j] = ethr_eV[i,j] / e3d_array[i,j] 
                    n3d_array[i,j] = self.n_array(zz[i,j], e3d_array[i,j])

        kernel = b3d_array * b3d_array * n3d_array * p_kernel(1. - b3d_array) * e3d_array

        result = simps(kernel, np.log(e3d_array), axis = 2)

        if 'gilmore' not in self._model:
            result *= (1. + zz) * (1. + zz) * (1. + zz)

        result[result == 0.] = np.ones(np.sum(result == 0.)) * 1e-40

        return np.squeeze((1. / (result * c.sigma_T.to('cm * cm').value * 0.75))*u.cm).to('Mpc').value
