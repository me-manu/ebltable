"""
Class to read EBL models of Kneiske & Dole 2010 and Franceschini et al. 2008

History of changes:
Version 1.0
- Created 22nd September 2011
Version 1.0.1
- added inverse tau function to return energy, 27nd September 2011
Version 1.0.2
- 11/7/2011: changed tau and inverse tau to scipy interpolation
Version 1.0.3
- 08/06/2012: Implemented array operation for optical depth calculation 

"""

__version__ = 1.02
__author__ = "M. Meyer // manuel.meyer@physik.uni-hamburg.de"

# ---- IMPORTS -----------------------------------------------#
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline as RBSpline
from scipy.interpolate import UnivariateSpline as USpline
import warnings
import os
# ------------------------------------------------------------#


class OptDepth(object):
    """
    Class to calculate attenuation of gamma-rays due to interaction with the EBL from EBL models.
    
    Important: if using the predefined model files, the path to the model files has to be set through the 
    environment variable EBL_FILE_PATH

    Arguments
    ---------
    z:		redshift, m-dim numpy array, given by model file
    logEGeV:	log energy in GeV, n-dim numpy array, given by model file, in mu m
    tau:	nxm - dim array with optical depth values, given by model file
    model:	string, model name

    tauSpline:	rectilinear spline over nuInu
    """
    def __init__(self,file_name='None',model = 'None'):
	"""
	Initiate Optical depth model class. 

	Parameters
	----------
	file_name:	string, full path to EBL model file, with first column with log(wavelength), 
			first row with redshift and nu I nu values otherwise. If none, model files are used
	model:		string, EBL model to use if file_name == None. Currently supported models are listed in Notes Section
	path:		string, if environment variable EBL_FILE_PATH is not set, this path will be used.

	Returns
	-------
	Nothing
	"""

	self.z = np.array([])
	self.logEGeV = np.array([])
	self.tau = np.array([])
	self.model = 'kneiske'
	self.tauSpline = None
	if file_name == 'None' and model == 'None':
	    return
	else:
	    self.readfile(file_name = file_name, model = model)
	return

    def readfile(self, file_name='None',model = 'kneiske', path='/afs/desy.de/user/m/meyerm/projects/blazars/EBLmodelFiles/'):
	"""
	Read in Model file.

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
		finke		Finke et al. (2012)		http://www.phy.ohiou.edu/~finke/EBL/
		dominguez	Dominguez et al. (2011)
		inuoe		Inuoe et al. (2013)		http://www.slac.stanford.edu/~yinoue/Download.html
		gilmore		Gilmore et al. (2012)		(fiducial model)
	"""
	self.model = model

	try:
	    ebl_file_path = os.environ['EBL_FILE_PATH']
	except KeyError:
	    warnings.warn("The EBL File environment variable is not set! Using {0} as path instead.".format(path), RuntimeWarning)
	    ebl_file_path = path

	if model == 'kneiske' or model == 'dominguez' or model == 'finke':
	    if file_name == 'None':
		if model == 'kneiske':
		    file_name = os.path.join(ebl_file_path , 'tau_ebl_cmb_kneiske.dat')
		if model == 'dominguez':
		    file_name = os.path.join(ebl_file_path , 'tau_dominguez10.dat')
		if model == 'finke':
		    file_name = os.path.join(ebl_file_path , 'tau_modelC_Finke.txt')

	    data = np.loadtxt(file_name)
	    self.z = data[0,1:]
	    self.tau = data[1:,1:]
	    if model == 'kneiske':
		self.logEGeV = data[1:,0]
	    else:
		self.logEGeV = np.log10(data[1:,0]*1e3)

	elif model == 'franceschini':
	    if file_name == 'None':
		file_name = os.path.join(ebl_file_path , 'tau_fran08.dat')
	    data = np.loadtxt(file_name,usecols=(0,2))
	    self.logEGeV = np.log10(data[0:50,0]*1e3)
	    self.tau = np.zeros((len(self.logEGeV),len(data[:,1])/len(self.logEGeV)))
	    self.z = np.zeros((len(data[:,1])/len(self.logEGeV),))
	    for i in range(len(data[:,1])/len(self.logEGeV)):
		self.tau[:,i] = data[i*50:i*50+50,1]
		self.z[i] += 1e-3*(i+1.)
	elif model == 'inoue':
	    if file_name == 'None':
		file_name = os.path.join(ebl_file_path , 'tau_gg_baseline.dat')
	    data = np.loadtxt(file_name)
	    self.z = data[0,1:]
	    self.tau = data[1:,1:]
	    self.logEGeV = np.log10(data[1:,0]*1e3)
	elif model == 'gilmore':
	    if file_name == 'None':
		file_name = os.path.join(ebl_file_path , 'opdep_fiducial.dat')
	    data = np.loadtxt(file_name)
	    self.z = data[0,1:]
	    self.tau = data[1:,1:]
	    self.logEGeV = np.log10(data[1:,0]/1e3)
	else:
	    raise ValueError("Unknown EBL model chosen!")
	# Add zeros 
	#self.z = np.insert(self.z,0,0.)
	#self.tau = np.insert(self.tau,0,0.,axis=1)
	self.tauSpline = RBSpline(self.logEGeV,self.z,self.tau,kx=2,ky=2)
	return

    def opt_depth(self,z,E):
	"""
	Return optical depth for redshift z and Engergy (TeV) from BSpline Interpolation

	Parameter
	---------
	z:	float, redshift
	E:	float, energy in TeV

	Returns
	-------
	Optical depth from given model (float)
	"""
	if z < self.z[0]:
	    result = self.tauSpline(math.log10(E) + 3.,self.z[0])[0,0]/self.z[0] * z
	else:
	    result = self.tauSpline(math.log10(E) + 3.,z)[0,0]
	if result < 0.:
	    return 1e-20
	return result

    def opt_depth_array(self,z,E):
	"""
	Returns optical depth for redshift z and Engergy (TeV) from BSpline Interpolation for z,E arrays

	Parameters
	----------
	z: redshift
	    scalar or N-dim numpy array
	E: Energy in TeV
	    scalar or M-dim numpy array

	Returns
	-------
	(N x M)-np.array with corresponding optical depth values, or a scalar

	Notes
	-----
	if any z < self.z (from interpolation table), self.z[0] is used and RuntimeWarning is issued.
	This overestimates optical depth!

	"""
	if np.isscalar(E) and np.isscalar(z):
	    return self.opt_depth(z,E)
	elif np.isscalar(E):
	    E = np.array([E])
	elif np.isscalar(z):
	    z = np.array([z])

        if np.any(z < self.z[0]): warnings.warn(
	    "Warning: a z value is below interpolation range, zmin = {0}".format(self.z[0]), RuntimeWarning)

	result = np.zeros((z.shape[0],E.shape[0]))
	tt = np.zeros((z.shape[0],E.shape[0]))

	args_z = np.argsort(z)
	args_E = np.argsort(E)

	# Spline interpolation requires sorted lists
	tt[args_z,:] = self.tauSpline(np.log10(np.sort(E)*1e3),np.sort(z)).transpose()	
	result[:,args_E] = tt

	return result

    def opt_depth_Inverse(self,z,tau):
	"""
	Return log10(Energy/GeV) for redshift z and optical depth tau from BSpline Interpolation

	Parameter
	---------
	z:	float, redshift
	tau:	float, optical depth

	Returns
	-------
	energy, float, log10(E/GeV) 
	"""
	Enew = USpline(self.tauSpline(self.logEGeV,z)[:,0],self.logEGeV, s = 0, k = 2, ext = 'extrapolate')
	return Enew(tau)

    def opt_depth_Ebin(self,z,Ebin,func,params,Esteps = 50):
	"""
	Compute average optical depth within an energy bin assuming a specific spectral shape

	Parameters
	----------
	z:	float, redshift
	Ebin:	n-dim array with Energy bin boundaries in TeV
	func:	function for spectrum, needs to be of the form func(params,Energy [TeV]), needs to except 2xn dim arrays
	params:	parameters that are past to func

	kwargs
	------
	Esteps: int, number of energy integration steps, default: 50

	Returns
	-------
	(n-1)-dim array with average tau values for each energy bin.

	Notes
	-----
	Any energy dispersion is neglected.
	"""
	# design a 2d matrix with energy integration steps
	for i,E in enumerate(Ebin):
	    if not i:
		logE_array	= np.linspace(np.log(E),np.log(Ebin[i+1]),Esteps)
		t_array		= self.opt_depth_array(z,np.exp(logE_array))[0]
	    elif i == len(Ebin) - 1:
		break
	    else:
		logE		= np.linspace(np.log(E),np.log(Ebin[i+1]),Esteps)
		logE_array	= np.vstack((logE_array,logE))
		t_array		= np.vstack((t_array,self.opt_depth_array(z,np.exp(logE))[0]))
	# return averaged tau value
	return	simps(func(np.exp(logE_array),**params) * t_array * np.exp(logE_array), logE_array, axis = 1) / \
		simps(func(np.exp(logE_array),**params) * np.exp(logE_array), logE_array, axis = 1)

    def clear(self):
	self.z = np.array([])
	self.logEGeV = np.array([])
	self.tau = np.array([])
	self.model = 'kneiske'
	self.tauSpline = None
