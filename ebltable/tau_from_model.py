# ---- IMPORTS -----------------------------------------------#
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline as USpline
from .interpolate import GridInterpolator
import os
# ------------------------------------------------------------#


class OptDepth(GridInterpolator):
    """
    Class to calculate attenuation of gamma-rays due to interaction 
    with the EBL from EBL models.
    """
    
    def __init__(self, z, EGeV, tau, kx=1, ky=1, **kwargs):
        """
        Initiate Optical depth model class. 

        Parameters
        ----------
        z: array-like
            source redshift, m-dimensional

        EGeV: array-like
            Energies in GeV, n-dimensional

        tau: array-like
            n x m array with optical depth values

        kx: int
            order of interpolation spline along energy axis, default: 1

        ky: int
            order of interpolation spline along energy axis, default: 1

        kwargs: dict
            Additional kwargs passed to `~scipy.interpolate.RectBivariateSpline`
        """
        super(OptDepth, self).__init__(EGeV, z, tau, logx=True, kx=kx, ky=ky, **kwargs)

    @staticmethod
    def readmodel(model, kx=1, ky=1, pad_zeros=True):
        """
        Read in an EBL model from an EBL model file

        Parameters
        ----------
        model: str
            EBL model to use.
            Currently supported models are listed in Notes Section
        pad_zeros: bool
            if true, pad tau array with zeros for z=0
        kx: int
            Spline order in x direction
        ky: int
            Spline order in y direction

        Notes
        -----
        Supported EBL models:
        Name:                Publication:
        franceschini         Franceschini et al. (2008), http://www.astro.unipd.it/background/
        franceschini2017     Franceschini et al. (2017)
        saldana-lopez        Saldana-Lopez et al. (2021) https://www.ucm.es/blazars/ebl
        saldana-lopez-upper  Saldana-Lopez et al. (2021) upper uncertainty,  https://www.ucm.es/blazars/ebl
        saldana-lopez-lower  Saldana-Lopez et al. (2021) upper uncertainty,  https://www.ucm.es/blazars/ebl
        kneiske              Kneiske & Dole (2010)
        finke                Finke et al.(2010)                http://www.phy.ohiou.edu/~finke/EBL/
        dominguez            Dominguez et al. (2011)
        dominguez-upper      Dominguez et al. (2011) upper uncertainty
        dominguez-lower      Dominguez et al. (2011) lower uncertainty
        inoue                Inuoe et al. (2013), (baseline) http://www.slac.stanford.edu/~yinoue/Download.html
        inoue-low-pop3       Inuoe et al. (2013), (low pop 3) http://www.slac.stanford.edu/~yinoue/Download.html
        inoue-up-pop3        Inuoe et al. (2013), (up pop 3) http://www.slac.stanford.edu/~yinoue/Download.html
        gilmore              Gilmore et al. (2012) (fiducial model)
        gilmore-fixed        Gilmore et al. (2012) (fixed model)
        """
        ebl_file_path = os.path.join(os.path.split(__file__)[0], 'data/')

        if model == 'kneiske' or model.find('dominguez') >= 0 or model == 'finke' \
            or model == 'franceschini2017' or model.find('saldana') >= 0.:

            if model == 'kneiske':
                file_name = os.path.join(ebl_file_path, 'tau_ebl_cmb_kneiske.dat')
            elif model == 'dominguez':
                file_name = os.path.join(ebl_file_path, 'tau_dominguez11_cta.out')
            elif model == 'dominguez-upper':
                file_name = os.path.join(ebl_file_path, 'tau_upper_dominguez11_cta.out')
            elif model == 'dominguez-lower':
                file_name = os.path.join(ebl_file_path, 'tau_lower_dominguez11_cta.out')
            elif model == 'finke':
                file_name = os.path.join(ebl_file_path, 'tau_modelC_Finke.txt')
            elif model == 'franceschini2017':
                file_name = os.path.join(ebl_file_path, 'tau_fran17.dat')
            elif model == 'saldana-lopez':
                file_name = os.path.join(ebl_file_path, 'tau_saldana-lopez21.out')
            elif model == 'saldana-lopez-upper':
                file_name = os.path.join(ebl_file_path, 'tau_high_saldana-lopez21.out')
            elif model == 'saldana-lopez-lower':
                file_name = os.path.join(ebl_file_path, 'tau_low_saldana-lopez21.out')
            else:
                raise ValueError("Unknown EBL model chosen!")

            data = np.loadtxt(file_name)
            z = data[0,1:]
            tau = data[1:,1:]
            if model == 'kneiske':
                EGeV = np.power(10.,data[1:,0])
            else:
                EGeV = data[1:,0]*1e3

        elif model == 'franceschini':
            file_name = os.path.join(ebl_file_path, 'tau_fran08.dat')

            data = np.loadtxt(file_name, usecols=(0, 2))
            EGeV = data[0:50, 0]*1e3
            tau = np.zeros((len(EGeV),int(len(data[:,1])/len(EGeV))))
            z = np.zeros(int(len(data[:,1])/len(EGeV)))
            for i in range(z.size):
                tau[:,i] = data[i*50:i*50+50,1]
                z[i] += 1e-3*(i+1.)

        elif model.find('inoue') >= 0:
            if model == 'inoue':
                file_name = os.path.join(ebl_file_path, 'tau_gg_baseline.dat')
            elif model == 'inoue-low-pop3':
                file_name = os.path.join(ebl_file_path, 'tau_gg_low_pop3.dat')
            elif model == 'inoue-up-pop3':
                file_name = os.path.join(ebl_file_path, 'tau_gg_up_pop3.dat')
            else:
                raise ValueError("Unknown EBL model chosen!")
            data = np.loadtxt(file_name)
            z = data[0,1:]
            tau = data[1:,1:]
            EGeV = data[1:,0]*1e3

        elif model.find('gilmore') >= 0:
            if model == 'gilmore':
                file_name = os.path.join(ebl_file_path, 'opdep_fiducial.dat')
            elif model == 'gilmore-fixed':
                file_name = os.path.join(ebl_file_path, 'opdep_fixed.dat')
            else:
                raise ValueError("Unknown EBL model chosen!")
            data = np.loadtxt(file_name)
            z = data[0, 1:]
            tau = data[1:, 1:]
            EGeV = data[1:, 0]/1e3
        else:
            raise ValueError("Unknown EBL model chosen!")

        # insert z = 0 and tau = 0 for all E (only done in Finke model)
        # in order to get interpolation below z_min right
        if not model == 'finke' and pad_zeros:
            z = np.insert(z, 0, 0.)
            tau = np.hstack([np.zeros(EGeV.shape[0])[:, np.newaxis], tau])

        return OptDepth(z, EGeV, tau, kx=kx, ky=ky)

    @staticmethod
    def readascii(file_name, kx=1, ky=1, **kwargs):
        """
        Read in an optical depth model file from an arbritrary file.

        Parameters
        ----------
        file_name: str, 
            full path to optical depth model file, 
            with a (n+1) x (m+1) dimensional table.
            The zeroth column contains the energy values in Energy (GeV), 
            first row contains the redshift values. 
            The remaining values are the tau values. 
            The [0,0] entry will be ignored.

        kx: int
            order of interpolation spline along energy axis, default: 1

        ky: int
            order of interpolation spline along energy axis, default: 1

        kwargs: dict
            Additional kwargs passed to `~scipy.interpolate.RectBivariateSpline`
        """
        EGeV, z, tau = GridInterpolator._read_ascii(file_name)
        return OptDepth(z, EGeV, tau, kx=kx, ky=ky, **kwargs)

    @staticmethod
    def readfits(file_name,
                 hdu_tau_vs_z='TAU_VS_Z',
                 hdu_energies='ENERGIES',
                 zcol='REDSHIFT',
                 taucol='OPT_DEPTH',
                 ecol='ENERGY',
                 kx=1, ky=1,
                 **kwargs
                 ):
        """
        Read opacities from a fits file using the astropy.io module

        Parameters
        ----------
        filename: str, 
            full path to fits file containing the opacities, redshifts, and energies

        kwargs
        ------
        hdu_tau_vs_z: str, optional,
            name of hdu that contains `~astropy.Table` with redshifts and tau values

        hdu_energies: str, optional,
            name of hdu that contains `~astropy.Table` with energies

        zcol: str, optional,
            name of column of `~astropy.Table` with redshift values

        taucol: str, optional,
            name of column of `~astropy.Table` with optical depth values

        ecol: str, optional,
            name of column of `~astropy.Table` with energy values

        kx: int
            order of interpolation spline along energy axis, default: 1

        ky: int
            order of interpolation spline along energy axis, default: 1

        kwargs: dict
            Additional kwargs passed to `~scipy.interpolate.RectBivariateSpline`
        """

        EGeV, z, tau = GridInterpolator._read_fits(file_name,
                                                   hdu_name_grid=hdu_tau_vs_z,
                                                   hdu_name_x=hdu_energies,
                                                   xcol_name=ecol,
                                                   ycol_name=zcol,
                                                   Zcol_name=taucol,
                                                   xtarget_unit="GeV")

        return OptDepth(z, EGeV, tau, kx=kx, ky=ky, **kwargs)

    def writefits(self, filename, z, ETeV, overwrite=True):
        """
        Write optical depth to a fits file using 
        the astropy table environment. 

        Parameters
        ----------
        filename: str,
            full file path for output fits file

        z: array-like
            source redshift, m-dimensional

        ETeV: array-like
            Energies in TeV, n-dimensional

        overwrite: bool
            Overwrite existing file.
        """

        self._write_fits(filename, ETeV * 1e3, z,
                         hdu_name_grid="TAU_VS_Z",
                         hdu_name_x="ENERGIES",
                         xunit="TeV",
                         xcol_name="ENERGY",
                         ycol_name="REDSHIFT",
                         Zcol_name="OPT_DEPTH",
                         xtarget_unit="GeV", overwrite=overwrite)

    def opt_depth(self, z, ETeV):
        """
        Returns optical depth for redshift z and Engergy (TeV) from BSpline Interpolation for z,E arrays

        Parameters
        ----------
        z: array-like
            source redshift, m-dimensional

        ETeV: array-like
            Energies in TeV, n-dimensional

        Returns
        -------
        (m x n) `~numpy.ndarray` with corresponding optical depth values.
        If z or E are scalars, the corresponding axis will be squeezed.

        """
        result = self.evaluate(ETeV * 1e3, z)
        return result

    def opt_depth_inverse(self, z, tau):
        """
        Return Energy in GeV for redshift z and optical depth tau from interpolation

        Parameter
        ---------
        z: float, 
            redshift

        tau: float, 
            optical depth

        Returns
        -------
        float, energy in GeV
        """

        tau_array = self._spline(self._x, z)[:, 0]

        mask = np.concatenate([[True], np.diff(tau_array) > 0.])

        while not np.all(np.diff(tau_array[mask]) > 0.) and np.sum(mask):
            new_mask = np.full(mask.size, False)
            new_mask[mask] = np.concatenate([[True], np.diff(tau_array[mask]) > 0.])
            mask = new_mask

        if not np.sum(mask):
            raise ValueError("Could not interpolate tau vs E")

        Enew = USpline(tau_array[mask], self._x[mask],
                       s=0, k=1, ext='extrapolate')

        return np.power(10., Enew(tau))

    def opt_depth_Ebin(self, z, Ebin, func, params, Esteps=50):
        """
        Compute average optical depth within an energy bin assuming a specific spectral shape

        Parameters
        ----------
        z: float, 
            redshift
        Ebin: array-like
            Energies of bin bounds in TeV, n-dimensional
        func: function pointer
            Spectrum, needs to be of the form func(Energy [TeV], **params), 
            needs to except 2xn dim arrays
        params: dict,
            parameters that are past to func

        kwargs
        ------
        Esteps: int, 
                number of energy integration steps, default: 50

        Returns
        -------
        (n-1)-dim `~numpy.ndarray` with average tau values for each energy bin.

        Notes
        -----
        Any energy dispersion is neglected.
        """
        # design a 2d matrix with energy integration steps
        logE_array = []
        t_array = []
        for i,E in enumerate(Ebin):
            if i == len(Ebin) - 1:
                break
            logE_array.append(np.linspace(np.log(E), np.log(Ebin[i+1]), Esteps))
            t_array.append(self.opt_depth(z, np.exp(logE_array[-1])))

        logE_array = np.array(logE_array)
        t_array = np.array(t_array)
        # return averaged tau value
        return simps(func(np.exp(logE_array), **params) * t_array * np.exp(logE_array), logE_array, axis=1) / \
                simps(func(np.exp(logE_array), **params) * np.exp(logE_array), logE_array, axis=1)
