"""
Class to read EBL models of Kneiske & Dole 2010 and Franceschini et al. 2008
"""

# ---- IMPORTS -----------------------------------------------#
import numpy as np
import os
from scipy.interpolate import RectBivariateSpline as RBSpline
from scipy.interpolate import UnivariateSpline as USpline
from astropy.io import fits
from astropy.table import Table,Column 
from scipy.integrate import simps
import warnings
from os.path import join
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck15 as cosmo
from scipy.special import spence # equals gsl_sf_dilog(1-z)

# ------------------------------------------------------------#
def Pkernel(x):
    """Kernel function from Biteau & Williams (2015), Eq. (7)"""

    m = (x < 0.) & (x >= 1.)
    x[x < 0.] = np.zeros(np.sum(x < 0.))
    x[x >= 1.] = np.zeros(np.sum(x >= 1.))
    x = np.sqrt(x)

    result = np.log(2.) * np.log(2.)  - np.pi *np.pi / 6. \
	    + 2. * spence(0.5 + 0.5 * x) - (x + x*x*x) / (1. - x*x) \
	    + (np.log(1. + x) - 2. * np.log(2.)) * np.log(1. - x) \
	    + 0.5 * (np.log(1. - x) * np.log(1. - x) - np.log(1. + x) * np.log(1. + x)) \
	    + 0.5 * (1. + x*x*x*x) / (1. - x*x) * (np.log(1. + x) - np.log(1. - x))
    result[x <= 0.] = np.zeros(np.sum(x <= 0.))
    return result

class EBL(object):
    """
    Class to calculate EBL intensities from EBL models.
    
    Important: if using the predefined model files, the path to the model files has to be set through the 
    environment variable EBL_FILE_PATH

    Arguments
    ---------
    z:		redshift, m-dim numpy array, given by model file
    loglmu:	log wavelength, n-dim numpy array, given by model file, in mu m
    nuInu:	nxm - dim array with EBL intensity in nW m^-2 sr^-1, given by model file
    """

    def __init__(self, z, lmu, nuInu, kx = 2, ky = 2):
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

	{options}

	kx: int
	    order of interpolation spline along energy axis, default: 2
	ky: int
	    order of interpolation spline along energy axis, default: 2
	"""
	self._z = np.array(z)
	self._loglmu= np.log10(lmu)
	self._nuInu= np.log10(nuInu)
	self.__eblSpline = RBSpline(self._loglmu,self._z,self._nuInu,kx=kx,ky=ky)
	return

    @property
    def z(self):
	return self._z

    @z.setter
    def z(self,z,kx = 2, ky = 2):
	self._z = z
	self.__eblSpline = RBSpline(self._loglnu,self._z,self._nuInu,kx=kx,ky=ky)
	return 

    @property
    def loglmu(self):
	return self._loglmu

    @loglmu.setter
    def loglmu(self,lmu,kx = 2, ky = 2):
	self._loglmu = np.log10(lmu)
	self.__eblSpline = RBSpline(self._loglnu,self._z,self._nuInu,kx=kx,ky=ky)
	return 

    @property
    def nuInu(self):
	return self._nuInu

    @nuInu.setter
    def nuInu(self,nuInu,kx = 2, ky = 2):
	self._nuInu = np.log10(nuInu)
	self.__eblSpline = RBSpline(self._loglnu,self._z,self._nuInu,kx=kx,ky=ky)
	return 

    @staticmethod
    def readmodel(model = 'dominguez'):
	"""
	Read in an EBL model from an EBL model file

	Parameters
	----------
	model:		str, 
			EBL model to use.
			Currently supported models are listed in Notes Section

	Notes
	-----
	Supported EBL models:
		Name:		Publication:
		franceschini	Franceschini et al. (2008)	http://www.astro.unipd.it/background/
		kneiske		Kneiske & Dole (2010)
		dominguez	Dominguez et al. (2011)
		dominguez-upper	Dominguez et al. (2011) upper uncertainty
		dominguez-lower	Dominguez et al. (2011) lower uncertainty
		inuoe           Inuoe et al. (2013)	 	(baseline) http://www.slac.stanford.edu/~yinoue/Download.html
		inuoe-low-pop3  Inuoe et al. (2013)		(low pop 3) http://www.slac.stanford.edu/~yinoue/Download.html
		inuoe-up-pop3   baseline  Inuoe et al. (2013)   (up pop 3) http://www.slac.stanford.edu/~yinoue/Download.html
		inuoe		Inuoe et al. (2013)		http://www.slac.stanford.edu/~yinoue/Download.html
		gilmore		Gilmore et al. (2012)		(fiducial model)
		gilmore-fixed   Gilmore et al. (2012)		(fixed model)
		finke		Finke et al. (2012)		(model C) http://www.phy.ohiou.edu/~finke/EBL/
		cuba		Haardt & Madua (2012)		http://www.ucolick.org/~pmadau/CUBA/HOME.html
	"""
	ebl_file_path = os.path.join(os.path.split(__file__)[0],'data/')

	if model == 'kneiske':
	    file_name = join(ebl_file_path , 'ebl_nuFnu_tanja.dat')
	elif model == 'franceschini':
	    file_name = join(ebl_file_path , 'ebl_franceschini.dat')
	elif model == 'dominguez':
	    file_name = join(ebl_file_path , 'ebl_dominguez11.out')
	elif model == 'dominguez-upper':
	    file_name = join(ebl_file_path , 'ebl_upper_uncertainties_dominguez11.out')
	elif model == 'dominguez-lower':
	    file_name = join(ebl_file_path , 'ebl_lower_uncertainties_dominguez11.out')
	elif model == 'inoue':
	    file_name = join(ebl_file_path , 'EBL_z_0_baseline.dat')
	    #file_name = join(ebl_file_path , 'EBL_proper_baseline.dat')
	elif model == 'inoue-low-pop3':
	    file_name = join(ebl_file_path , 'EBL_z_0_low_pop3.dat')
	    #file_name = join(ebl_file_path , 'EBL_proper_low_pop3.dat')
	elif model == 'inoue-up-pop3':
	    file_name = join(ebl_file_path , 'EBL_z_0_up_pop3.dat')
	    #file_name = join(ebl_file_path , 'EBL_proper_up_pop3.dat')
	elif model == 'gilmore':
	    file_name = join(ebl_file_path , 'eblflux_fiducial.dat')
	elif model == 'gilmore-fixed':
	    file_name = join(ebl_file_path , 'eblflux_fixed.dat')
	elif model == 'cuba':
	    file_name = join(ebl_file_path , 'CUBA_UVB.dat')
	elif model == 'finke':
	    file_name = join(ebl_file_path , 'ebl_modelC_Finke.txt')
	else:
	    raise ValueError("Unknown EBL model chosen!")

	data = np.loadtxt(file_name)
	if model.find('inoue') >= 0:
	    z = np.array([0.])
	    #z = data[0,1:]
	    #nuInu = data[:,1]
	    lmu = data[:,0]
	    nuInu = np.array([data[:,1]]).T
	    raise ValueError('Inoue models not correctly implemented at the moment, choose another model')

	elif model.find('gilmore') >= 0:
	    z = data[0,1:]
	    lmu = data[1:,0] * 1e-4 # convert from Angstrom to micro meter
	    nuInu = data[1:,1:]			
	    nuInu[nuInu == 0.] = 1e-20 * np.ones(np.sum(nuInu == 0.))
	    
	    # convert from ergs/s/cm^2/Ang/sr to nW/m^2/sr
	    nuInu = (nuInu.T * data[1:,0]).T * 1e4 * 1e-7 * 1e9	

	elif model == 'cuba':
	    z = data[0,1:-1]
	    lmu = data[1:,0] * 1e-4
	    nuInu = data[1:,1:-1]

	    # replace zeros by 1e-40
	    idx = np.where(data[1:,1:-1] == 0.)
	    nuInu[idx] = np.ones(np.sum(nuInu == 0.)) * 1e-20

	    # in erg / cm^2 / s / sr
	    nuInu = (nuInu.T * c.c.value / (lmu * 1e-6)).T	
	    nuInu *= 1e6	# in nW / m^2 /  sr

	    # check where lmu is not strictly increasing
	    idx = np.where(np.diff(lmu) == 0.)
	    for i in idx[0]:
		lmu[i+1] = (lmu[i + 2] + lmu[i]) / 2.

	else:
	    z = data[0,1:]
	    lmu = data[1:,0]
	    nuInu = data[1:,1:]
	    if model == 'finke': 
		lmu = lmu[::-1] * 1e-4
		nuInu = nuInu[::-1]

	return EBL(z,lmu,nuInu)

    @staticmethod
    def readascii(file_name):
	"""
	Read in an EBL model file from an arbritrary file.

	Parameters
	----------
	file_name:	str, 
			full path to EBL photon density model file, 
			with a (n+1) x (m+1) dimensional table.
			The zeroth column contains the wavelength values in mu meter, 
			first row contains the redshift values. 
			The remaining values are the EBL photon density values in nW / m^2 / sr. 
			The [0,0] entry will be ignored.
	"""
	data = np.loadtxt(file_name)
	z = data[0,1:]
	nuInu = data[1:,1:]
	lmu = data[1:,0]
	return EBL(z, lmu, nuInu)

    @staticmethod
    def readfits(file_name,
		hdu_nuInu_vs_z= 'NUINU_VS_Z',
		hdu_wavelength='WAVELENGTHS',
		zcol='REDSHIFT',
		eblcol='EBL_DENS',
		lcol='WAVELENGTH'):
	"""
	Read EBL photon density from a fits file using the astropy.io module

	Parameters
	----------
	filename: str, 
		full path to fits file containing the opacities, redshifts, and energies

	{options} 

	hdu_nuInu_vs_z: str, optional,
		name of hdu that contains `~astropy.Table` with redshifts and tau values
	hdu_wavelengths: str, optional,
		name of hdu that contains `~astropy.Table` with wavelegnths
	zcol: str, optional,
		name of column of `~astropy.Table` with redshift values
	eblcol: str, optional,
		name of column of `~astropy.Table` with EBL density values
	lcol: str, optional,
		name of column of `~astropy.Table` with wavelength values
	"""
	t = Table.read(file_name, hdu = hdu_nuInu_vs_z)
	z = t[zcol].data
	ebl = t[eblcol].data
	t2 = Table.read(file_name, hdu = hdu_wavelength)
	lmu = t2[lcol].data * t2[lcol].unit
	return EBL(z,lmu.to(u.micrometer).value,ebl.T)

    def writefits(self,filename, z,lmu):
	"""
	Write optical depth to a fits file using 
	the astropy table environment. 

	Parameters
	----------
	filename: str,
	     full file path for output fits file

	z: `~numpy.ndarray` or list
	    source redshift, m-dimensional

	lmu: `~numpy.ndarray` or list
	    wavelenghts in micrometer, n-dimensional
	"""
	t = Table([z,self.ebl_array(z,lmu)], names = ('REDSHIFT', 'EBL_DENS'))
	t2 = Table()
	t2['WAVELENGTH'] = Column(lmu, unit = 'micrometer')

	hdulist = fits.HDUList([fits.PrimaryHDU(),fits.table_to_hdu(t),fits.table_to_hdu(t2)])

	hdulist[1].name = 'NUINU_VS_Z'
	hdulist[2].name = 'WAVELENGTHS'

	hdulist.writeto(filename, overwrite = True)
	return

    def ebl_array(self,z,lmu):
	"""
	Returns EBL intensity in nuInu [nW / m^2 / sr] 
	for redshift z and wavelegth l (micron) from BSpline Interpolation

	Parameters
	----------
	z: `~numpy.ndarray` or list
	    source redshift, m-dimensional

	lmu: `~numpy.ndarray` or list
	    wavelenghts in micrometer, n-dimensional

	Returns
	-------
	(n x m)-dim `~numpy.ndarray` with corresponding (nu I nu) values

	Notes
	-----
	if any z < self._z[0] (from interpolation table), 
	self._z[0] is used and RuntimeWarning is issued.

	"""
	if np.isscalar(lmu):
	    lmu = np.array([lmu])
	elif type(lmu) == list:
	    ETeV = np.array(lmu)
	if np.isscalar(z):
	    z = np.array([z])
	elif type(z) == list:
	    z = np.array(z)

        if np.any(z < self._z[0]): warnings.warn(
	    "Warning: a z value is below interpolation range, zmin = {0:.2f}".format(self._z[0]), 
	    RuntimeWarning)

	result	= np.zeros((z.shape[0],lmu.shape[0]))
	tt	= np.zeros((z.shape[0],lmu.shape[0]))

	args_z = np.argsort(z)
	args_l = np.argsort(lmu)

	tt[args_z,:]		= self.__eblSpline(np.log10(np.sort(lmu)),
				    np.sort(z)).T # Spline interpolation requires sorted lists
	result[:,args_l]	= tt

	return np.power(10., result)

    def n_array(self,z,EeV):
	"""
	Returns EBL photon density in [1 / cm^3 / eV] for redshift z and energy from BSpline Interpolation

	Parameters
	----------
	z: `~numpy.ndarray` or list or tuple
	    source redshift, n-dimensional

	EeV: `~numpy.ndarray` or list or tuple
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
	elif type(EeV) == list or type(EeV) == tuple:
	    EeV = np.array(EeV)
	if np.isscalar(z):
	    z = np.array([z])
	elif type(z) == list or type(z) == tuple:
	    z = np.array(z)

	# convert energy in eV to wavelength in micron
	l	=  (c.h * c.c / (EeV * u.eV).to(u.J)).to(u.um).value #SI_h * SI_c / EeV / SI_e  * 1e6	
	# convert energy in J
	e_J	= (EeV *u.eV).to(u.J).value

	n = self.ebl_array(z,l)
	# convert nuInu to photon density in 1 / J / m^3
	n = 4.* np.pi / c.c.value / e_J**2. * n  * 1e-9
	# convert photon density in 1 / eV / cm^3 and return
	return n * c.e.value * 1e-6

    def ebl_int(self,z,lmin = 0.01,lmax=1e3,steps = 50):
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

	{options}

	steps: int
	    number of steps for simps integration

	Returns
	-------
	Float with integrated nuInu value

	"""
	logl = np.linspace(np.log10(lmin),np.log10(lmax),steps)
	lnl  = np.log(np.linspace(lmin,lmax,steps))
	ln_Il = np.log(10.) * (self.ebl_array(z,10.**logl)) 	# note: nuInu = lambda I lambda
	result = simps(ln_Il,lnl)
	return result

    def optical_depth(self,z0,ETeV,
			#OmegaM = cosmo.Om0, OmegaL = 1. - cosmo.Om0, 
			OmegaM = 0.7, OmegaL = 0.3, 
			H0 = cosmo.H0.value, 
			steps_z = 50,
			steps_e = 50,
			egamma_LIV = True,
			LIV_scale = 0., nLIV=1):
	"""
	calculates mean free path for gamma-ray energy ETeV at redshift z

	Parameters
	----------
	z0:	float 
		source redshift
	ETeV:   `~numpy.ndarray` or list or tuple
		Energies in TeV, n-dimensional

	{options}

	H0 :	    float,
	            Hubble constant in km / Mpc / s. Default: Planck 2015 result
	OmegaM :    float,
	            critical density of matter at z=0. Default: Planck 2015 result
	OmegaL :    float,
	            critical density of dark energy. Default: 0

	steps_z :   int
		    intergration steps for redshift (default : 50)
	steps_e:    int
		    number of integration steps for integration over EBL density (default: 60)
		
	LIV_scale:  float 
		    Lorentz invariance violation parameter (quantum gravity scale), 
		    if 0. (default), do not include LIV
	nLIV:	    int 
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
	elif type(ETeV) == list or type(ETeV) == tuple:
	    ETeV = np.array(ETeV)

	H0 = (H0 * cosmo.H0.unit).to('1 / s').value # convert from km / Mpc / s to 1 / s

	z_array = np.linspace(0.,z0,steps_z)
	result = self.mean_free_path(z_array,ETeV,
					    LIV_scale = LIV_scale,
					    nLIV = nLIV, 
					    egamma_LIV = egamma_LIV,
					    steps_e = steps_e)
	zz = np.meshgrid(z_array, ETeV)[0]
	result = 1. / (result * u.Mpc).to(u.cm).value # this is in cm^-1
	result *= 1./ ( (1. + zz ) * np.sqrt((1.+ zz )**3. * OmegaM + OmegaL) )	# dt / dz for a flat universe
	result = simps(result,zz, axis = 1)
	return result * c.c.to('cm / s').value /  H0


	#return  result * 1e9 * yr2sec * c.c.to('cm / s').value /  H0
	return  
    def mean_free_path(self,z,ETeV,
    			steps_e = 50,
			LIV_scale = 0., nLIV=1, 
			egamma_LIV=True):
	"""
	calculates mean free path in Mpc for gamma-ray energy ETeV at redshift z

	z:      `~numpy.ndarray` or list or tuple
		redshift, n-dimensional
	ETeV:   `~numpy.ndarray` or list or tuple
		Energies in TeV, m-dimensional

	{options}

	steps_e:    int
		    number of integration steps for integration over EBL density (default: 60)
	LIV_scale:  float 
		    Lorentz invariance violation parameter (quantum gravity scale), 
		    if 0. (default), do not include LIV
	nLIV:	    int 
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
	see Dwek & Krennrich 2013 or Mirizzi & Montanino 2009.
	For kernel see Biteau & Williams 2015
	"""
	if np.isscalar(ETeV):
	    ETeV = np.array([ETeV])
	elif type(ETeV) == list or type(ETeV) == tuple:
	    ETeV = np.array(ETeV)
	if np.isscalar(z):
	    z = np.array([z])
	elif type(z) == list or type(z) == tuple:
	    z = np.array(z)

	# planck mass in eV
	Mpl_eV = (np.sqrt(c.hbar * c.c / c.G) * c.c**2.).to('eV').value
	# electron mass in eV
	m_e_eV = (c.m_e * c.c**2.).to('eV').value

	# max energy of EBL template in eV
	emax_eV		= (c.h * c.c / (10.**np.min(self.loglmu) * u.um)).to('eV').value 

	# defines the effective energy scale for the LIV modification
	if LIV_scale:
	    ELIV_eV = 4.*m_e_eV* m_e_eV * (LIV_scale * Mpl_eV)**nLIV
	    if egamma_LIV:
		ELIV_eV /= 1. - 2.**(-nLIV)
	    ELIV_eV = ELIV_eV ** (1./(2.+nLIV))

	# make z, ETeV to 2d arrays
	zz, EE_TeV = np.meshgrid(z,ETeV) # m x n dimensional
	EEzz_eV = EE_TeV * 1e12 * (1. + zz)
	ethr_eV = m_e_eV * m_e_eV / EEzz_eV
	if LIV_scale:
	    ethr_eV	*= 1.+ (EEzz_eV/ELIV_eV)**(nLIV+2.)

	b3d_array = np.ones(ethr_eV.shape + (steps_e,)) * 1e-40
	e3d_array = np.ones(ethr_eV.shape + (steps_e,)) * 1e-40
	n3d_array = np.ones(ethr_eV.shape + (steps_e,)) * 1e-40

	for i in range(ethr_eV.shape[0]): # loop over ETeV dimension
	    for j in range(ethr_eV.shape[1]): #loop over z dimension
		if ethr_eV[i,j] < emax_eV:
		    e3d_array[i,j] =  np.logspace(np.log10(ethr_eV[i,j]), np.log10(emax_eV), steps_e)
		    b3d_array[i,j] = ethr_eV[i,j] / e3d_array[i,j] 
		    #b3d_array.mask[i,j] = np.zeros(steps_e, dtype = np.bool)
		    n3d_array[i,j] = self.n_array(zz[i,j], e3d_array[i,j])

	kernel = b3d_array * b3d_array * n3d_array * Pkernel(1. - b3d_array) * e3d_array
	result = simps(kernel, np.log(e3d_array), axis = 2)
	return np.squeeze((1. / (result * c.sigma_T.to('cm * cm').value * 0.75))*u.cm).to('Mpc').value
