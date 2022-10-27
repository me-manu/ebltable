from __future__ import absolute_import, division, print_function
import numpy as np
import warnings
from collections import Iterable
from astropy.table import Table, Column
from astropy.io import fits
from astropy.units import Unit
from scipy.interpolate import RectBivariateSpline


class GridInterpolator(object):
    """
    Base class for 2D grid interpolation
    """

    def __init__(self, x, y, Z, logx=False, logy=False, logZ=False, kx=1, ky=1, **kwargs):
        """
        Initialize the class.

        Parameters
        ----------
        x: array-like
            Array with values of x axis of grid, n-dimensional

        y: array-like
            Array with values of y axis of grid, m-dimensional

        z: array-like
            Array with grid (z) values, n x m-dimensional

        logx: bool
            Use logarithmic interpolation over x axis. Default: False

        logy: bool
            Use logarithmic interpolation over y axis. Default: False

        logZ: bool
            Use logarithmic interpolation over Z axis. Default: False

        kx: int
            Order of spline interpolation in x direction. Default: 1

        ky: int
            Order of spline interpolation in y direction. Default: 1

        kwargs: dict
            Additional kwargs passed to `~scipy.interpolate.RectBivariateSpline`
        """
        kx = kwargs.pop('kx', kx)
        ky = kwargs.pop('ky', ky)
        self._logx = logx
        self._logy = logy
        self._logZ = logZ

        if logx:
            x[x == 0.] = 1e-40
            self._x = np.log10(x)
        else:
            self._x = x

        if logy:
            y[y == 0.] = 1e-40
            self._y = np.log10(y)
        else:
            self._y = y

        if logZ:
            Z[Z == 0.] = 1e-40
            self._Z = np.log10(Z)
        else:
            self._Z = Z

        self._spline = RectBivariateSpline(self._x, self._y, self._Z, kx=kx, ky=ky, **kwargs)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def Z(self):
        return self._Z

    @staticmethod
    def _read_ascii(file_name):
        """
        Read in a model file from an arbitrary file.

        Parameters
        ----------
        file_name: str,
            full path to optical depth model file,
            with a (n+1) x (m+1) dimensional table.
            The zeroth column contains the x values,
            first row contains the y values.
            The remaining values are the Z values of the grid.
            The [0,0] entry will be ignored.

        Returns
        -------
        tuple with x, y and Z values
        """
        data = np.loadtxt(file_name)

        x = data[1:, 0]
        y = data[0, 1:]

        Z = data[1:, 1:]

        return x, y, Z


    @staticmethod
    def _read_fits(file_name, hdu_name_grid, hdu_name_x,
                   xcol_name, ycol_name, Zcol_name,
                   xtarget_unit):
        """
        Read in a model file from an arbitrary file.

        Parameters
        ----------
        file_name: str,
            full path to fits file containing the grid values

        hdu_name_grid: str
            Name of the HDU extension containing the Grid values Z and y axis values

        hdu_name_x: str,
            Name of the HDU extension containing the x axis values

        xcol_name: str,
            name of x column in hdu_name_x extension

        ycol_name: str,
            name of y column in hdu_name_grid extension

        Zcol_name: str,
            name of Z column in hdu_name_grid extension

        xtarget_unit: str,
            name of target unit for x values

        TODO: this is not working properly yet!

        Returns
        -------
        tuple with x, y and Z values
        """
        t = Table.read(file_name, hdu=hdu_name_grid)
        y = t[ycol_name].data
        Z = t[Zcol_name].data

        t2 = Table.read(file_name, hdu=hdu_name_x)
        x = t2[xcol_name].data * t2[xcol_name].unit

        return x.to(xtarget_unit).value, y, Z.T


    def _write_fits(self, filename, x, y, hdu_name_grid, hdu_name_x,
                    xunit, xcol_name, ycol_name, Zcol_name, xtarget_unit="", overwrite=True):
        """
        Write Z values to a fits file using
        the astropy table environment.

        Parameters
        ----------
        filename: str,
            full file path for output fits file

        x: array-like
            x values for interpolation

        y: array-like
            y values for interpolation

        hdu_name_grid: str
            Name of the HDU extension containing the Grid values Z and y axis values

        hdu_name_x: str,
            Name of the HDU extension containing the x axis values

        xunit: str,
            name of unit for x values

        xcol_name: str,
            name of x column in hdu_name_x extension

        ycol_name: str,
            name of y column in hdu_name_grid extension

        Zcol_name: str,
            name of Z column in hdu_name_grid extension

        xtarget_unit: str
            name of unit of x values

        overwrite: bool
            Overwrite existing file.

        TODO: this is not working properly yet!
        """
        t = Table([y, self.evaluate(x, y)],
                  names=(ycol_name, Zcol_name))

        t2 = Table()
        t2[xcol_name] = Column(x * Unit(xunit).to(xtarget_unit),  unit=xtarget_unit)

        hdulist = fits.HDUList([fits.PrimaryHDU(),
                                fits.table_to_hdu(t),
                                fits.table_to_hdu(t2)])

        hdulist[1].name = hdu_name_grid
        hdulist[2].name = hdu_name_x

        hdulist.writeto(filename, overwrite=overwrite)
        return

    def evaluate(self, x, y):
        """
        Evaluate Spline for some x and y values

        Parameters
        ----------
        x: array-like
            x coordinates for evaluation, n-dimensional

        y: array-like
            y coordinates for evaluation, m-dimensional

        Returns
        -------
        Interpolated Z values for input x and y values (m x n dimensional).
        If n or m are equal to one, drop this axis.
        """
        if np.isscalar(x):
            x = np.array([x])
        elif x is Iterable:
            x = np.array(x)

        if np.isscalar(y):
            y = np.array([y])
        elif y is Iterable:
            y = np.array(y)

        if np.any(y < self._y[0]):
            warnings.warn(f"Warning: a y value is below interpolation range, y min = {self._y[0]:.2f}",
                          RuntimeWarning)

        result = np.zeros((y.shape[0], x.shape[0]))
        tt = np.zeros((y.shape[0], x.shape[0]))

        args_x = np.argsort(x)
        args_y = np.argsort(y)

        # Spline interpolation requires sorted lists
        # alternative would be to calculate the spline with grid=False
        # but this takes longer in my tests
        if self._logx:
            x = np.log10(x)

        if self._logy:
            y = np.log10(y)

        tt[args_y, :] = self._spline(np.sort(x), np.sort(y)).transpose()
        result[:, args_x] = tt

        if self._logZ:
            result = np.power(10., result)

        return np.squeeze(result)
