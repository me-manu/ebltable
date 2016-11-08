"""
Class to read EBL models of Kneiske & Dole 2010 and Franceschini et al. 2008

History of changes:
Version 0.01
- Created 29th October 2012
Version 0.02
- 7th March: added inoue z = 0 EBL
"""

__version__ = 0.02
__author__ = "M. Meyer // manuel.meyer@physik.uni-hamburg.de"

# ---- IMPORTS -----------------------------------------------#
import numpy as np
import math
import logging
import os
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBSpline
from scipy.integrate import simps
import warnings
from numpy import linspace
from numpy import meshgrid 
from numpy import zeros
from numpy import transpose
from numpy import swapaxes 
from numpy.ma import masked_array
from numpy import dstack
from numpy import log,sqrt 
from eblstud.misc.constants import *
from os.path import join
import time
# ------------------------------------------------------------#

class EBL(object):
    """
    Class to calculate EBL intensities from EBL models.
    
    Important: if using the predefined model files, the path to the model files has to be set through the 
    environment variable EBL_FILE_PATH

    Arguments
    ---------
    z:		redshift, m-dim numpy array, given by model file
    logl:	log wavelength, n-dim numpy array, given by model file, in mu m
    nuInu:	nxm - dim array with EBL intensity in nW m^-2 sr^-1, given by model file
    model:	string, model name

    eblSpline:	rectilinear spline over nuInu

    steps_mu 	int, steps for tau - integration in mu = cos(angle between photon momenta)
    steps_e 	int, steps for tau - integration in redshift
    steps_z 	int, steps for tau - integration in energy
    """

    def __init__(self,file_name='None',model = 'kneiske', path='/afs/desy.de/user/m/meyerm/projects/blazars/EBLmodelFiles/'):
	"""
	Initiate EBL model class. 

	Parameters
	----------
	file_name:	string, full path to EBL model file, with first column with log(wavelength), 
			first row with redshift and nu I nu values otherwise. If none, model files are used
	model:		string, EBL model to use if file_name == None. Currently supported models are listed in Notes Section
	path:		string, if environment variable EBL_FILE_PATH is not set, this path will be used.

	Returns
	-------
	Nothing

	Notes
	-----
	Supported EBL models:
		Name:		Publication:
		franceschini	Franceschini et al. (2008)	http://www.astro.unipd.it/background/
		kneiske		Kneiske & Dole (2010)
		dominguez	Dominguez et al. (2011)
		inuoe		Inuoe et al. (2013)		http://www.slac.stanford.edu/~yinoue/Download.html
		gilmore		Gilmore et al. (2012)		(fiducial model)
		finke		Finke et al. (2012)		http://www.phy.ohiou.edu/~finke/EBL/
	"""
	self.z		= np.array([])		#redshift
	self.logl	= np.array([])		#log (wavelength / micron)
	self.nuInu	= np.array([])		#log (nu I_nu / [nW / Hz / sr])
	self.model	= model		#model


	if file_name == 'None':
	    try:
		ebl_file_path = os.environ['EBL_FILE_PATH']
	    except KeyError:
		logging.warning("The EBL File environment variable is not set! Using {0} as path instead.".format(path))
		ebl_file_path = path

	    if model == 'kneiske':
		file_name = join(ebl_file_path , 'ebl_nuFnu_tanja.dat')
	    elif model == 'franceschini':
		file_name = join(ebl_file_path , 'ebl_franceschini.dat')
	    elif model == 'dominguez':
		file_name = join(ebl_file_path , 'ebl_dominguez.dat')
	    elif model == 'inoue':
		file_name = join(ebl_file_path , 'EBL_z_0_baseline.dat')
		logging.warning("Inoue model is only provided for z = 0!")
	    elif model == 'gilmore':
		file_name = join(ebl_file_path , 'eblflux_fiducial.dat')
	    elif model == 'cuba':
		file_name = join(ebl_file_path , 'CUBA_UVB.dat')
	    elif model == 'finke':
		file_name = join(ebl_file_path , 'ebl_modelC_Finke.txt')
	    else:
		raise ValueError("Unknown EBL model chosen!")

	    data = np.loadtxt(file_name)
	    if model == 'inoue':
		self.z = np.array([0.])
		self.logl = np.log10(data[:,0])
		self.nuInu = np.log10(data[:,1])
		self.eblSpline = interp1d(self.logl,self.nuInu)
		return
	    elif model == 'gilmore':
		self.z = data[0,1:]
		self.logl = np.log10(data[1:,0] * 1e-4)			# convert from Angstrom to micro meter
		self.nuInu = data[1:,1:]			
		self.nuInu[self.nuInu == 0.] = 1e-20 * np.ones(np.sum(self.nuInu == 0.))
		self.nuInu = (self.nuInu.T * data[1:,0]).T * 1e4 * 1e-7 * 1e9		# convert from ergs/s/cm^2/Ang/sr to nW/m^2/sr
		self.nuInu = np.log10(self.nuInu)
	    elif model == 'cuba':
		self.z = data[0,1:-1]
		self.logl = np.log10(data[1:,0] * 1e-4)
		self.nuInu = data[1:,1:-1]
		# replace zeros by 1e-40
		idx = np.where(data[1:,1:-1] == 0.)
		self.nuInu[idx] = np.ones(np.sum(self.nuInu == 0.)) * 1e-20
		self.nuInu = np.log10(self.nuInu.transpose() * SI_c / (10.**self.logl * 1e-6)).transpose()	# in erg / cm^2 / s / sr
		self.nuInu += 6	# in nW / m^2 /  sr

		# check where logl is not strictly increasing
		idx = np.where(np.diff(self.logl) == 0.)
		for i in idx[0]:
		    self.logl[i+1] = (self.logl[i + 2] + self.logl[i]) / 2.
	    else:
		self.z = data[0,1:]
		self.logl = np.log10(data[1:,0])
		self.nuInu = np.log10(data[1:,1:])
		if model == 'finke': 
		    self.logl = self.logl[::-1] - 4.
		    self.nuInu = self.nuInu[::-1]
	else:
	    data	= np.loadtxt(file_name)
	    self.z	= data[0,1:]
	    self.logl	= np.log10(data[1:,0])
	    self.nuInu	= np.log10(data[1:,1:])

	self.eblSpline = RBSpline(self.logl,self.z,self.nuInu,kx=2,ky=2)	# reutrns log10(nuInu) for log10(lambda)

	self.steps_mu		= 50	# steps for integration
	self.steps_e		= 50	# steps for integration
	self.steps_z		= 50	# steps for integration

	return

    def optical_depth(self,z0,E_TeV,LIV = 0.,h = h, OmegaM = OmegaM, OmegaL = OmegaL):
	"""
	calculates mean free path for gamma-ray energy E_TeV at redshift z

	Parameters
	----------
	z:	float, redshift
	E_TeV:	n-dim array, gamma ray energy in TeV
	LIV:	float, Lorentz invariance violation parameter (quantum gravity scale), 
		assumed for linear violation of disp. relation and only photons affected (see Jacob & Piran 2008)

	Returns
	-------
	(N dim)-np.array with corresponding mean free path values in Mpc

	Notes
	-----
	See Dwek & Krennrich 2013 and Mirizzi & Montanino 2009
	"""
	if np.isscalar(E_TeV):
	    E_TeV = np.array([E_TeV])

	z_array = linspace(0.,z0,self.steps_z)
	result	= zeros((self.steps_mu,E_TeV.shape[0]))

	for i,z in enumerate(z_array):
	    result[i] = 1. / self.mean_free_path(z,E_TeV,LIV = LIV) / Mpc2cm	# this is in cm^-1
	    result[i] *=1./ ( (1. + z ) * math.sqrt((1.+ z)**3. * OmegaM + OmegaL) )	# dt / dz for a flat universe

	result = simps(result,z_array, axis = 0)

	return  result * 1e9 * yr2sec * CGS_c /  H0 

    def mean_free_path(self,z,E_TeV,LIV = 0.):
	"""
	calculates mean free path for gamma-ray energy E_TeV at redshift z

	Parameters
	----------
	z:	float, redshift
	E_TeV:	n-dim array, gamma ray energy in TeV
	LIV:	float, Lorentz invariance violation parameter (quantum gravity scale), 
		assumed for linear violation of disp. relation and only photons affected (see Jacob & Piran 2008)

	Returns
	-------
	(N dim)-np.array with corresponding mean free path values in Mpc

	Notes
	-----
	See Dwek & Krennrich 2013 and Mirizzi & Montanino 2009
	"""
	if np.isscalar(E_TeV):
	    E_TeV = np.array([E_TeV])
	    

	mu_array	= linspace(-1.,1.,self.steps_mu, endpoint = False)

	e_max		= SI_h * SI_c / 10.**np.min(self.logl) * 1e6 / SI_e	# max energy of EBL template in eV
	e_min		= SI_h * SI_c / 10.**np.max(self.logl) * 1e6 / SI_e	# min energy of EBL template in eV

	logl_array	= linspace(np.max(self.logl), np.min(self.logl - 5.), self.steps_e)
	e_array		= SI_h * SI_c / 10.**logl_array * 1e6 / SI_e

	n		= self.ebl_array(z,10.**logl_array)[0] / (e_array * e_array)
	nn 		= transpose(meshgrid(n,E_TeV)[0])		# rows: constant e, columns: decreasing e / increasing l

	result = zeros((mu_array.shape[0], E_TeV.shape[0]))

#	print np.min(e_array), 'here'

	for i,m in enumerate(mu_array):
	    if not LIV:
		ethr	= 2. * M_E_EV * M_E_EV / (E_TeV*1e12 * (1. + z) * (1. - m))
	    else:
		ethr	= 2. * M_E_EV * M_E_EV / (E_TeV*1e12 * (1. + z) * (1. - m)) \
			    + 0.25 * (E_TeV * (1. + z) / (LIV * 1.22e16)) * E_TeV * 1e12 * (1. + z)

# change integration interval for each mu value:
	    e_array	= 10.**np.linspace(np.log10(np.min(ethr)),4.,self.steps_e)
	    logl_array	= SI_h * SI_c / e_array / SI_e *1e6

	    n		= self.ebl_array(z,logl_array)[0] / (e_array * e_array)
	    nn 		= transpose(meshgrid(n,E_TeV)[0])		# rows: constant e, columns: decreasing e / increasing l

	    ee,EE	= meshgrid(ethr,e_array)	# ethr, with const E_TeV each column
	    						# e with const e in each row
	    b		= masked_array(sqrt(1. - (ee / EE)),mask = EE < ee)	# beta, with E_TeV changing in col,
										# e changing with rows, masked array
	    bb		= b*b
	    sigma	= (1. - bb) * (2. * b * (bb - 2.) + (3. - bb * bb) * log( (1. + b) / (1. - b) ) )
	    sigma 	*= nn * (1. + z)*(1. + z)*(1. + z)

	    result[i] = simps(sigma * EE, np.log(EE), axis = 0)		#integrate over rows

	mm	= meshgrid(E_TeV,mu_array)[1]
	result *= (1. - mm) * 0.5
	result = simps(result,mm, axis = 0) * 3. / 16. * CGS_tcs * 2.62e-4	# last factor to convert nuInu to 1 / ev / cm^3

	return 1. / result / Mpc2cm

    def n_array(self,z,e):
	"""
	Returns EBL photon density in [1 / cm^3 / eV] for redshift z and energy e (eV) from BSpline Interpolation

	Parameters
	----------
	z: redshift
	    scalar or N-dim numpy array
	e: energy in eV 
	    scalar or M-dim numpy array

	Returns
	-------
	(N x M)-np.array with corresponding photon density values

	Notes
	-----
	if any z < self.z (from interpolation table), self.z[0] is used and RuntimeWarning is issued.
	"""
	if np.isscalar(e):
	    e = np.array([e])
	if np.isscalar(z):
	    z = np.array([z])

	# convert energy in eV to wavelength in micron
	l	=  SI_h * SI_c / e / SI_e  * 1e6	
	# convert energy in J
	e_J	= e * SI_e

	n = self.ebl_array(z,l)
	# convert nuInu to photon density in 1 / J / m^3
	n = 4.*PI / SI_c / e_J**2. * n  * 1e-9
	# convert photon density in 1 / eV / cm^3 and return
	return n * SI_e * 1e-6

    def ebl_array(self,z,l):
	"""
	Returns EBL intensity in nuInu [W / m^2 / sr] for redshift z and wavelegth l (micron) from BSpline Interpolation

	Parameters
	----------
	z: redshift
	    scalar or N-dim numpy array
	l: wavelength
	    scalar or M-dim numpy array

	Returns
	-------
	(N x M)-np.array with corresponding (nu I nu) values

	Notes
	-----
	if any z < self.z (from interpolation table), self.z[0] is used and RuntimeWarning is issued.

	"""
	if np.isscalar(l):
	    l = np.array([l])
	if np.isscalar(z):
	    z = np.array([z])

	if self.model == 'inoue':
	    logging.warning("Inoue model is only provided for z = 0!")
	    return np.array([10.**self.eblSpline(np.log10(l))])

        if np.any(z < self.z[0]): warnings.warn("Warning: a z value is below interpolation range, zmin = {0}".format(self.z[0]), RuntimeWarning)

	result	= np.zeros((z.shape[0],l.shape[0]))
	tt	= np.zeros((z.shape[0],l.shape[0]))

	args_z = np.argsort(z)
	args_l = np.argsort(l)

	tt[args_z,:]		= self.eblSpline(np.log10(np.sort(l)),np.sort(z)).transpose()	# Spline interpolation requires sorted lists
	result[:,args_l]	= tt

	return 10.**result

    def ebl_int(self,z,lmin = 10.**-0.7,lmax=10.**3.,steps = 50):
	"""
	Returns integrated EBL intensity in I [nW / m^2 / sr] for redshift z between wavelegth lmin and lmax (micron) 

	Parameters
	----------
	z: redshift
	    scalar 
	lmin: wavelength
	    scalar
	lmax: wavelength
	    scalar
	steps: number of steps for simps integration
	    integer

	Returns
	-------
	Scalar with corresponding I values

	Notes
	-----
	if z < self.z (from interpolation table), self.z[0] is used and RuntimeWarning is issued.
	"""
	steps = 50
	logl = np.linspace(np.log10(lmin),np.log10(lmax),steps)
	lnl  = np.log(np.linspace(lmin,lmax,steps))
	ln_Il = np.log(10.) * (self.ebl_array(z,10.**logl)[0]) 	# note: nuInu = lambda I lambda
	result = simps(ln_Il,lnl)
	return result

    def clear(self):
	self.z = np.array([])
	self.logl= np.array([])
	self.nuInu= np.array([])
	self.model = 'None'
	self.eblSpline = None
