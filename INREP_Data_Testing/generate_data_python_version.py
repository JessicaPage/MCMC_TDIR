import sys
from mpmath import binomial,log,pi,ceil
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import fftconvolve,cosine
#from mpmath import *
import time
from scipy.stats import norm, multivariate_normal
from astropy import constants as const
from scipy.fft import next_fast_len
from lisainstrument import Instrument
#from pytdi.michelson import X2



i=Instrument(size=345600,dt=0.25,orbits='static',clock_asds=None,aafilter=None,physics_upsampling=1)

#i.disable_all_noises(but='laser')

i.disable_dopplers()
i.disable_clock_noises()
#i.disable_pathlength_noises()
i.disable_ranging_noises()


i.plot_mprs(output=None, skip=500)

mosas_order = i.MOSAS
print('mosas order')
print(mosas_order)
print('delays')
print(i.mprs)

science_measurements = np.array([i.isc_carrier_fluctuations[mosa]/i.central_freq for mosa in mosas_order])
TM_measurements = [i.tm_carrier_fluctuations[mosa]/i.central_freq for mosa in mosas_order]
ref_measurements = [i.ref_carrier_fluctuations[mosa]/i.central_freq for mosa in mosas_order]

first_concat = np.concatenate((science_measurements,ref_measurements))
all_measurements = np.concatenate((first_concat,TM_measurements))

np.savetxt('LISA_Instrument_RR_more_noise.dat',all_measurements.T,header = 's21 s32 s13 s31 s23 s12 tau21 tau32 tau13 tau31 tau23 tau12 eps21 eps32 eps13 eps31 eps23 eps12')