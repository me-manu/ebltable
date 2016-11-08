# Example script to calculate the gamma-ray attenuation 
# given a certain EBL model and a source at a redshift z

# ------ Imports --------------- #
from ebltable.tau_from_model import OptDepth as OD
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------ #

# initialize the optical depth class
# the model keyword specifies the EBL model to be used.
#Supported EBL models:
#	Name:		Publication:
#	franceschini	Franceschini et al. (2008)	http://www.astro.unipd.it/background/
#	kneiske		Kneiske & Dole (2010)
#	dominguez	Dominguez et al. (2011)
#	inuoe		Inuoe et al. (2013)		http://www.slac.stanford.edu/~yinoue/Download.html
#	gilmore		Gilmore et al. (2012)		(fiducial model)
# Note: make sure that you have set the environment variable EBL_FILE_PATH
# by e.g. 
# export EBL_FILE_PATH=/path/to/repro/ebl/ebl_model_files
tau = OD(model = 'franceschini')

# Source redshift
z	= 0.2
# array with energies in TeV
ETeV = np.logspace(-1,1,50)

# calculating the optical depth for a redshift z and TeV energies
# this returns a two dimensional array with dimensions 
# of the redshift array times dimensions of the energy array. 
# Since z is a scalar here, it will return a 1 x 50 dim array.
t = tau.opt_depth_array(z,ETeV)

# calculate the energy in TeV for which we reach an optical depth of 1:
tauLim = 1.
ETeV_tEq1 = 10.**tau.opt_depth_Inverse(z,tauLim) / 1e3
# note that this function call only supports scalar input

# PLOT THE RESULT
# plot the expected attenuation
ax = plt.subplot(111)
plt.loglog(ETeV, np.exp(-1. * t[0]), lw = 2., ls = '-', label = '$z = {0:.2f}$'.format(z))
plt.axvline(ETeV_tEq1, ls = '--', label = r'$\tau = {0:.2f}$'.format(tauLim))
plt.xlabel('Energy (TeV)')
plt.ylabel('Attenuation')
plt.legend(loc = 3)
plt.show()
