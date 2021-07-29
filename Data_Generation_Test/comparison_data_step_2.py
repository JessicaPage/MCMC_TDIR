import numpy as np
import matplotlib.pyplot as plt
import math
from mpmath import *
from mpmath import binomial,log,pi,ceil
import sys
import time
from astropy import constants as const
from scipy.signal import fftconvolve, cosine



def next_two_power(n):

	return int(pow(2, ceil(log(n, 2))))

def filter(M,f_max,fs):

	f_c = f_max/fs

	h = np.empty(M+1)
	#symmetric point set to this to avoid divide by zero
	h[int(M/2)] = 2.0*math.pi*f_c
	#then either side is equation as normal as given in dspguide.com ch 16
	h[:int(M/2)] = (0.42-0.5*np.cos(2.0*np.pi*np.arange(int(M/2))/M)+0.08*np.cos(4.0*np.pi*np.arange(int(M/2))/M))*np.sin(2.0*np.pi*f_c*(np.arange(int(M/2))-M/2))/(np.arange(int(M/2))-M/2)
	h[int(M/2+1):] = (0.42-0.5*np.cos(2.0*np.pi*np.arange(int(M/2+1),M+1)/M)+0.08*np.cos(4.0*np.pi*np.arange(int(M/2+1),M+1)/M))*np.sin(2.0*np.pi*f_c*(np.arange(int(M/2+1),M+1)-M/2))/(np.arange(int(M/2+1),M+1)-M/2)

	sum = np.sum(h)
	h = h/sum

	return h


#window for LaGrange function WITH MPMATH
def lagrange(n,N,D_here):

	D=D_here
	N = float(N)
	n = float(n) 
	t_D = 0.5*(N-1)+D
	if D < 1.e-12:
		window = 1.
	else:
		window = math.pi*N/(math.sin(math.pi*t_D))*binomial(t_D,N)*binomial((N-1),(n+(N-1)*0.5))
	return window


def S_y_proof_mass_new_frac_freq(f):
	pm_here =  np.power(3.0e-15,2)*(1+np.power(4.0e-4/f,2))*(1+np.power(f/8.0e-3,4))
	#*np.power(2*np.pi*f,-4)
	return pm_here*np.power(2*np.pi*f*const.c.value,-2)

def S_y_OMS_frac_freq(f):
	op_here =  np.power(1.5e-11,2)*np.power(2*np.pi*f/const.c.value,2)*(1+np.power(2.0e-3/f,4))
	return op_here


dur = 12*3600
f_samp = 4

f_min = 1e-4
f_max = 1.0

L_arm = 2.5e9

#simulated delay times in seconds rounded to 9 decimal places
L_1 = 8.339095192
L_1_p = 8.338994879
L_2 = 8.338867041
L_2_p = 8.339095192
L_3 = 8.338994879
L_3_p = 8.338867041


#........................................................................................
#............................Laser Frequency Noises......................................
#............................Band-passed, white noise from overlap-add method....................................

data_science_ref = np.genfromtxt('comparison_data_fs_1e3_N=101_LFN_only.dat',names=True)

s31_LFN = data_science_ref['s31_LFN']
s21_LFN = data_science_ref['s21_LFN']
s32_LFN = data_science_ref['s32_LFN']
s12_LFN = data_science_ref['s12_LFN']
s23_LFN = data_science_ref['s23_LFN']
s13_LFN = data_science_ref['s13_LFN']


tau31_LFN = data_science_ref['tau31_LFN']
tau21_LFN = data_science_ref['tau21_LFN']
tau12_LFN = data_science_ref['tau12_LFN']
tau32_LFN = data_science_ref['tau32_LFN']
tau23_LFN = data_science_ref['tau23_LFN']
tau13_LFN = data_science_ref['tau13_LFN']

eps31_LFN = data_science_ref['eps31_LFN']
eps21_LFN = data_science_ref['eps21_LFN']
eps12_LFN = data_science_ref['eps12_LFN']
eps32_LFN = data_science_ref['eps32_LFN']
eps23_LFN = data_science_ref['eps23_LFN']
eps13_LFN = data_science_ref['eps13_LFN']




#number between each sample for downsampling
m = len(s31_LFN)



#filter creation process
#low-pass portion
h_low = filter(m,f_max,f_samp)
#high-pass portion
h_high  = filter(m,f_min,f_samp)
#spectral inversion
h_high = -1*h_high
h_high[int(m/2)]+=1
#create band-reject by adding the two
h_ = h_low + h_high
#spectral inversion of band-reject to create band-pass
h_inv = -1*h_
h_inv[int(m/2)]+=1

#filter used in band-passing each pij
h = np.fft.rfft(h_inv)
f=np.fft.rfftfreq(m,1/f_samp)
if f[1] >= 1e-4:
	print('issues arise with frequency resolution and indexing. See page 855')
	sys.exit()


#........................................................................................
#............................Secondary Noises............................................
#........................................................................................

# 'OP' really means 'OMS' noise here


analytic_PM = S_y_proof_mass_new_frac_freq(f)
analytic_OP = S_y_OMS_frac_freq(f)



#first frequency bin is zero; divide by zero error. Next bin is < 1e-4 so makes no difference.
analytic_PM[0] = analytic_PM[1]
analytic_OP[0] = analytic_OP[1]



noise_PM_1 = h*(np.sqrt(analytic_PM/2)*(np.random.randn(len(analytic_PM)) + 1j*np.random.randn(len(analytic_PM))))
noise_OP_31 = h*(np.sqrt(analytic_OP/2)*(np.random.randn(len(analytic_OP)) + 1j*np.random.randn(len(analytic_OP))))

noise_PM_1_star = h*(np.sqrt(analytic_PM/2)*(np.random.randn(len(analytic_PM)) + 1j*np.random.randn(len(analytic_PM))))
noise_OP_21 = h*(np.sqrt(analytic_OP/2)*(np.random.randn(len(analytic_OP)) + 1j*np.random.randn(len(analytic_OP))))

noise_PM_2 = h*(np.sqrt(analytic_PM/2)*(np.random.randn(len(analytic_PM)) + 1j*np.random.randn(len(analytic_PM))))
noise_OP_32 = h*(np.sqrt(analytic_OP/2)*(np.random.randn(len(analytic_OP)) + 1j*np.random.randn(len(analytic_OP))))


noise_PM_2_star = h*(np.sqrt(analytic_PM/2)*(np.random.randn(len(analytic_PM)) + 1j*np.random.randn(len(analytic_PM))))
noise_OP_12 = h*(np.sqrt(analytic_OP/2)*(np.random.randn(len(analytic_OP)) + 1j*np.random.randn(len(analytic_OP))))

noise_PM_3 = h*(np.sqrt(analytic_PM/2)*(np.random.randn(len(analytic_PM)) + 1j*np.random.randn(len(analytic_PM))))
noise_OP_23 = h*(np.sqrt(analytic_OP/2)*(np.random.randn(len(analytic_OP)) + 1j*np.random.randn(len(analytic_OP))))

noise_PM_3_star = h*(np.sqrt(analytic_PM/2)*(np.random.randn(len(analytic_PM)) + 1j*np.random.randn(len(analytic_PM))))
noise_OP_13 = h*(np.sqrt(analytic_OP/2)*(np.random.randn(len(analytic_OP)) + 1j*np.random.randn(len(analytic_OP))))




#irfft
time_noise_PM_1 = np.fft.irfft(noise_PM_1,norm='ortho')
time_noise_OP_31 = np.fft.irfft(noise_OP_31,norm='ortho')

#irfft
time_noise_PM_1_star = np.fft.irfft(noise_PM_1_star,norm='ortho')
time_noise_OP_21 = np.fft.irfft(noise_OP_21,norm='ortho')


#irfft
time_noise_PM_2 = np.fft.irfft(noise_PM_2,norm='ortho')
time_noise_OP_32 = np.fft.irfft(noise_OP_32,norm='ortho')


#irfft
time_noise_PM_2_star = np.fft.irfft(noise_PM_2_star,norm='ortho')
time_noise_OP_12 = np.fft.irfft(noise_OP_12,norm='ortho')


	
#irfft
time_noise_PM_3 = np.fft.irfft(noise_PM_3,norm='ortho')
time_noise_OP_23 = np.fft.irfft(noise_OP_23,norm='ortho')


#irfft
time_noise_PM_3_star = np.fft.irfft(noise_PM_3_star,norm='ortho')
time_noise_OP_13 = np.fft.irfft(noise_OP_13,norm='ortho')


#............................................................
#create sij by subtracting delayed pij_k by appropriate pij
#also add PM and OP noises appropriately.
#............................................................


s31 = s31_LFN + time_noise_OP_31 
s21 = s21_LFN + time_noise_OP_21
s32 = s32_LFN + time_noise_OP_32 
s12 = s12_LFN + time_noise_OP_12
s23 = s23_LFN + time_noise_OP_23 
s13 = s13_LFN + time_noise_OP_13 

s31_noise = time_noise_OP_31 
s21_noise = time_noise_OP_21 
s32_noise = time_noise_OP_32 
s12_noise = time_noise_OP_12 
s23_noise = time_noise_OP_23 
s13_noise = time_noise_OP_13 

#............................................................
#create tau_ij by just subtracting
#............................................................

tau31 = tau31_LFN
tau21 = tau21_LFN
tau12 = tau12_LFN
tau32 = tau32_LFN
tau23 = tau23_LFN
tau13 = tau13_LFN


tau31_noise = np.zeros(len(tau31))
tau21_noise = np.zeros(len(tau31))
tau12_noise = np.zeros(len(tau31))
tau32_noise = np.zeros(len(tau31))
tau23_noise = np.zeros(len(tau31))
tau13_noise = np.zeros(len(tau31))




#............................................................
#create epsilon_ij by just subtracting
#............................................................

eps31 = eps31_LFN + 2*time_noise_PM_1
eps21 = eps21_LFN - 2*time_noise_PM_1_star
eps12 = eps12_LFN + 2*time_noise_PM_2
eps32 = eps32_LFN - 2*time_noise_PM_2_star
eps23 = eps23_LFN + 2*time_noise_PM_3
eps13 = eps13_LFN - 2*time_noise_PM_3_star


eps31_noise = 2*time_noise_PM_1
eps21_noise = -2*time_noise_PM_1_star
eps12_noise = 2*time_noise_PM_2
eps32_noise = -2*time_noise_PM_2_star
eps23_noise = 2*time_noise_PM_3
eps13_noise = -2*time_noise_PM_3_star




#............................................................
#save to file
#............................................................
np.savetxt('comparison_data_added_secondary_noise.dat',np.c_[s31, s21,s32, s12, s23, s13, tau31, tau21, tau12, tau32, tau23, tau13, eps31, eps21, eps12, eps32, eps23, eps13, s31_noise, s21_noise,s32_noise, s12_noise, s23_noise, s13_noise, tau31_noise, tau21_noise, tau12_noise, tau32_noise, tau23_noise, tau13_noise, eps31_noise, eps21_noise, eps12_noise, eps32_noise, eps23_noise, eps13_noise, s31_LFN, s21_LFN,s32_LFN, s12_LFN, s23_LFN, s13_LFN, tau31_LFN, tau21_LFN, tau12_LFN, tau32_LFN, tau23_LFN, tau13_LFN, eps31_LFN, eps21_LFN, eps12_LFN, eps32_LFN, eps23_LFN, eps13_LFN],header='s31 s21 s32 s12 s23 s13 tau31 tau21 tau12 tau32 tau23 tau13 eps31 eps21 eps12 eps32 eps23 eps13 s31_noise s21_noise s32_noise s12_noise s23_noise s13_noise tau31_noise tau21_noise tau12_noise tau32_noise tau23_noise tau13_noise eps31_noise eps21_noise eps12_noise eps32_noise eps23_noise eps13_noise s31_LFN s21_LFN s32_LFN s12_LFN s23_LFN s13_LFN tau31_LFN tau21_LFN tau12_LFN tau32_LFN tau23_LFN tau13_LFN eps31_LFN eps21_LFN eps12_LFN eps32_LFN eps23_LFN eps13_LFN')

