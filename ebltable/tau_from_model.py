"""
Class to read gamma-ray absorption from EBL models
"""

# ---- IMPORTS -----------------------------------------------#
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline as RBSpline
from scipy.interpolate import UnivariateSpline as USpline
from astropy.io import fits
from astropy.table import Table,Column
import astropy.units as u
import warnings
import os
# ------------------------------------------------------------#


class OptDepth(object):
    """
    Class to calculate attenuation of gamma-rays due to interaction 
    with the EBL from EBL models.
    
    Important: if using the predefined model files, 
    the path to the model files has to be set through the 
    environment variable EBL_FILE_PATH

    Arguments
    ---------
    z:		redshift, m-dim numpy array, given by model file
    logEGeV:	log10 energy in GeV, n-dim numpy array, given by model file
    tau:	nxm - dim array with optical depth values, given by model file
    """
    

    def __init__(self, z, EGeV, tau,kx = 2, ky = 2):
	"""
	Initiate Optical depth model class. 

	Parameters
	----------
	z: `~numpy.ndarray` or list
	    source redshift, m-dimensional

	EGeV: `~numpy.ndarray` or list
	    Energies in GeV, n-dimensional

	tau: `~numpy.ndarray` or list
	    n x m array with optical depth values

	{options}

	kx: int
	    order of interpolation spline along energy axis, default: 2
	ky: int
	    order of interpolation spline along energy axis, default: 2
	"""

	self._z = np.array(z)
	self._logEGeV = np.log10(EGeV)
	self._tau = np.array(tau)
	self.__tauSpline = RBSpline(self._logEGeV,self._z,self._tau,kx=kx,ky=ky)
	return

    @property
    def z(self):
	return self._z

    @z.setter
    def z(self,z,kx = 2, ky = 2):
	self._z = z
	self.__tauSpline = RBSpline(self._logEGeV,self._z,self._tau,kx=kx,ky=ky)
	return 

    @property
    def logEGeV(self):
	return self._logEGeV

    @logEGeV.setter
    def logEGeV(self,EGeV,kx = 2, ky = 2):
	self._logEGeV = np.log10(EGeV)
	self.__tauSpline = RBSpline(self._logEGeV,self._z,self._tau,kx=kx,ky=ky)
	return 

    @property
    def tau(self):
	return self._tau

    @tau.setter
    def tau(self,tau,kx = 2, ky = 2):
	self._tau = tau
	self.__tauSpline = RBSpline(self._logEGeV,self._z,self._tau,kx=kx,ky=ky)
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
		finke		Finke et al. (2012)		http://www.phy.ohiou.edu/~finke/EBL/
		dominguez	Dominguez et al. (2011)
		dominguez-upper	Dominguez et al. (2011) upper uncertainty
		dominguez-lower	Dominguez et al. (2011) lower uncertainty
		inuoe		Inuoe et al. (2013)		http://www.slac.stanford.edu/~yinoue/Download.html
		inuoe-low-pop3  Inuoe et al. (2013)		(low pop 3) http://www.slac.stanford.edu/~yinoue/Download.html
		inuoe-up-pop3   baseline  Inuoe et al. (2013)   (up pop 3) http://www.slac.stanford.edu/~yinoue/Download.html
		gilmore		Gilmore et al. (2012)		(fiducial model)
		gilmore-fixed   Gilmore et al. (2012)		(fixed model)
	"""
	ebl_file_path = os.path.join(os.path.split(__file__)[0],'data/')

	if model == 'kneiske' or model.find('dominguez') >= 0 or model == 'finke':
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
	    file_name = os.path.join(ebl_file_path , 'tau_fran08.dat')

	    data = np.loadtxt(file_name,usecols=(0,2))
	    EGeV = data[0:50,0]*1e3
	    tau = np.zeros((len(EGeV),len(data[:,1])/len(EGeV)))
	    z = np.zeros((len(data[:,1])/len(EGeV),))
	    for i in range(len(data[:,1])/len(EGeV)):
		tau[:,i] = data[i*50:i*50+50,1]
		z[i] += 1e-3*(i+1.)

	elif model.find('inoue') >= 0:
	    if model == 'inoue':
		file_name = os.path.join(ebl_file_path , 'tau_gg_baseline.dat')
	    elif model == 'inoue-low-pop3':
		file_name = os.path.join(ebl_file_path , 'tau_gg_low_pop3.dat')
	    elif model == 'inoue-up-pop3':
		file_name = os.path.join(ebl_file_path , 'tau_gg_up_pop3.dat')
	    else:
		raise ValueError("Unknown EBL model chosen!")
	    data = np.loadtxt(file_name)
	    z = data[0,1:]
	    tau = data[1:,1:]
	    EGeV = data[1:,0]*1e3

	elif model.find('gilmore') >= 0:
	    if model == 'gilmore':
		file_name = os.path.join(ebl_file_path , 'opdep_fiducial.dat')
	    elif model == 'gilmore-fixed':
		file_name = os.path.join(ebl_file_path , 'opdep_fixed.dat')
	    else:
		raise ValueError("Unknown EBL model chosen!")
	    data = np.loadtxt(file_name)
	    z = data[0,1:]
	    tau = data[1:,1:]
	    EGeV = data[1:,0]/1e3
	else:
	    raise ValueError("Unknown EBL model chosen!")

	return OptDepth(z,EGeV, tau)

    @staticmethod
    def readascii(file_name):
	"""
	Read in an EBL model file from an arbritrary file.

	Parameters
	----------
	file_name:	str, 
			full path to optical depth model file, 
			with a (n+1) x (m+1) dimensional table.
			The zeroth column contains the energy values in Energy (GeV), 
			first row contains the redshift values. 
			The remaining values are the tau values. 
			The [0,0] entry will be ignored.
	"""
	data = np.loadtxt(file_name)
	z = data[0,1:]
	tau = data[1:,1:]
	EGeV = data[1:,0]
	return OptDepth(z, EGeV, tau)

    @staticmethod
    def readfits(file_name,
		hdu_tau_vs_z= 'TAU_VS_Z',
		hdu_energies='ENERGIES',
		zcol='REDSHIFT',
		taucol='OPT_DEPTH',
		ecol='ENERGY'):
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
	"""
	t = Table.read(file_name, hdu = hdu_tau_vs_z)
	z = t[zcol].data
	tau = t[taucol].data
	t2 = Table.read(file_name, hdu = hdu_energies)
	EGeV = t2[ecol].data * t2[ecol].unit
	return OptDepth(z,EGeV.to(u.GeV).value,tau.T)

    def writefits(self,filename, z,ETeV):
	"""
	Write optical depth to a fits file using 
	the astropy table environment. 

	Parameters
	----------
	filename: str,
	     full file path for output fits file
	z: `~numpy.ndarray` or list
	    source redshift, m-dimensional

	ETeV: `~numpy.ndarray` or list
	    Energies in TeV, n-dimensional
	"""
	t = Table([z,self.opt_depth(z,ETeV)], names = ('REDSHIFT', 'OPT_DEPTH'))
	t2 = Table()
	t2['ENERGY'] = Column(ETeV * 1e3, unit = 'GeV')

	hdulist = fits.HDUList([fits.PrimaryHDU(),fits.table_to_hdu(t),fits.table_to_hdu(t2)])

	hdulist[1].name = 'TAU_VS_Z'
	hdulist[2].name = 'ENERGIES'

	hdulist.writeto(filename, overwrite = True)
	return

    def opt_depth(self,z,ETeV):
	"""
	Returns optical depth for redshift z and Engergy (TeV) from BSpline Interpolation for z,E arrays

	Parameters
	----------
	z: `~numpy.ndarray` or list
	    source redshift, m-dimensional

	ETeV: `~numpy.ndarray` or list
	    Energies in TeV, n-dimensional

	Returns
	-------
	(N x M) `~numpy.ndarray` with corresponding optical depth values.
	If z or E are scalars, the corresponding axis will be squezed.

	Notes
	-----
	if any z < self._z (from interpolation table), self._z[0] is used and RuntimeWarning is issued.
	This might overestimate the optical depth!

	"""
	if np.isscalar(ETeV):
	    ETeV = np.array([ETeV])
	elif type(ETeV) == list:
	    ETeV = np.array(ETeV)
	if np.isscalar(z):
	    z = np.array([z])
	elif type(z) == list:
	    z = np.array(z)

        if np.any(z < self._z[0]): warnings.warn(
	    "Warning: a z value is below interpolation range, zmin = {0:.2f}".format(self._z[0]), 
	    RuntimeWarning)

	result = np.zeros((z.shape[0],ETeV.shape[0]))
	tt = np.zeros((z.shape[0],ETeV.shape[0]))

	args_z = np.argsort(z)
	args_E = np.argsort(ETeV)

	# Spline interpolation requires sorted lists
	tt[args_z,:] = self.__tauSpline(np.log10(np.sort(ETeV)*1e3),np.sort(z)).transpose()	
	result[:,args_E] = tt

	return np.squeeze(result)

    def opt_depth_inverse(self,z,tau):
	"""
	Return Energy in GeV for redshift z and optical depth tau from BSpline Interpolation

	Parameter
	---------
	z:	float, 
		redshift
	tau:	float, 
		optical depth

	Returns
	-------
	float, energy in GeV
	"""
	Enew = USpline(self.__tauSpline(self._logEGeV,z)[:,0],self._logEGeV,
		s = 0, k = 1, ext = 'extrapolate')
	return np.power(10.,Enew(tau))

    def opt_depth_Ebin(self,z,Ebin,func,params,Esteps = 50):
	"""
	Compute average optical depth within an energy bin assuming a specific spectral shape

	Parameters
	----------
	z:	float, 
		redshift
	Ebin: 	`~numpy.ndarray` or list
		Energies of bin bounds in TeV, n-dimensional
	func:	function pointer
		Spectrum, needs to be of the form func(Energy [TeV], **params), 
		needs to except 2xn dim arrays
	params:	dict,
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
	    logE_array.append(np.linspace(np.log(E),np.log(Ebin[i+1]),Esteps))
	    t_array.append(self.opt_depth(z,np.exp(logE_array)))

	logE_array = np.array(logE_array)
	t_array = np.array(t_array)
	# return averaged tau value
	return	simps(func(np.exp(logE_array),**params) * t_array * np.exp(logE_array), logE_array, axis = 1) / \
		simps(func(np.exp(logE_array),**params) * np.exp(logE_array), logE_array, axis = 1)
