import numpy as np
import matplotlib.pyplot as plt
import math
from mpmath import *
from mpmath import binomial,log,pi,ceil
import sys
import time
from astropy import constants as const
from scipy.signal import fftconvolve, cosine

def cut_data(D_3,D_2,D_1,D_3_p,D_2_p,D_1_p,f_rate,length):

	D_2_val = int(round(f_rate*D_2))
	D_3_val = int(round(f_rate*D_3))
	D_1_val = int(round(f_rate*D_1))

	D_2_p_val = int(round(f_rate*D_2_p))
	D_3_p_val = int(round(f_rate*D_3_p))
	D_1_p_val = int(round(f_rate*D_1_p))

	filter_cut = int(round((number_n-1)))

	beg_ind = filter_cut+D_3_val+D_2_val+D_1_val+D_3_p_val+D_2_p_val+D_1_p_val
	#beg_ind = filter_cut
	end_ind = int(length-filter_cut-1)


	return beg_ind, end_ind

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



#...........................CREATING FILTERS.......................................
def filters_lagrange(delay):
	D = delay*f_samp
	i, d_frac = divmod(D,1)
	# delayed filters we're convolving with
	if d_frac >= 0.5:
		i+=1
		d_frac = -1*(1-d_frac)
	lagrange_filter = np.zeros(len(h_points))
	h_points_filter = h_points - int(i) +1
	indices = np.where(np.logical_and(h_points_filter>= int(-(number_n-1)/2.0),h_points_filter <= int((number_n-1)/2.0)))

	lagrange_filter[indices[0]] = [lagrange(int(n),number_n,d_frac)*np.sinc(int(n)-d_frac) for n in h_points_filter[indices[0]]]

	return lagrange_filter


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
f_samp = 1e3
f_s = 4

down_sample_factor = int(f_samp/f_s)

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




#number between each sample for downsampling
#down_sample_factor = int(f_samp/f_s)
m = int(dur*f_samp)


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


del h_low
del h_high
del h_
del h_inv

#........................................................................................
#............................Laser Frequency Noises......................................
#............................Band-passed, white noise....................................
length_f = len(h)


band_passed_indices = np.logical_and(f>=f_min,f<=f_max)
normalizeable_length = len(band_passed_indices)
#ratio = length_f/normalizeable_length
ratio = np.sqrt(down_sample_factor)

p12f = 1e-13/np.sqrt(2)*(np.random.randn(length_f) + 1j*np.random.randn(length_f))
p12bp = ratio*h*p12f
p12 = np.fft.irfft(p12bp,norm='ortho')


p13f = 1e-13/np.sqrt(2)*(np.random.randn(length_f) + 1j*np.random.randn(length_f))
p13bp = ratio*h*p13f
p13 = np.fft.irfft(p13bp,norm='ortho')


p23f = 1e-13/np.sqrt(2)*(np.random.randn(length_f) + 1j*np.random.randn(length_f))
p23bp = ratio*h*p23f
p23 = np.fft.irfft(p23bp,norm='ortho')


p21f = 1e-13/np.sqrt(2)*(np.random.randn(length_f) + 1j*np.random.randn(length_f))
p21bp = ratio*h*p21f
p21 = np.fft.irfft(p21bp,norm='ortho')


p31f = 1e-13/np.sqrt(2)*(np.random.randn(length_f) + 1j*np.random.randn(length_f))
p31bp = ratio*h*p31f
p31 = np.fft.irfft(p31bp,norm='ortho')


p32f = 1e-13/np.sqrt(2)*(np.random.randn(length_f) + 1j*np.random.randn(length_f))
p32bp = ratio*h*p32f
p32 = np.fft.irfft(p32bp,norm='ortho')



#............................................................
#creating delayed pij,k by filtering with scipy's fftconvolve
#............................................................

length = len(p32) # length of data

nearest_number = length

# number points in filter
if nearest_number%2 == 0:
	h_points = np.arange(-nearest_number/2.0,nearest_number/2.0,1)
else:
	h_points = np.arange(-(nearest_number-1)/2.0,(nearest_number-1)/2.0+1,1)

#lagrange filter length
number_n = 101

#ratio = down_sample_factor

p13_2 = fftconvolve(p13,filters_lagrange(L_2),'same')

p32_1 = fftconvolve(p32,filters_lagrange(L_1),'same')

p12_3p = fftconvolve(p12,filters_lagrange(L_3_p),'same')

p23_1p = fftconvolve(p23,filters_lagrange(L_1_p),'same')

p21_3 = fftconvolve(p21,filters_lagrange(L_3),'same')

p31_2p = fftconvolve(p31,filters_lagrange(L_2_p),'same')

beg_ind,end_ind = cut_data(L_3,L_2,L_1,L_3_p,L_2_p,L_1_p,f_samp,length)
#............................................................
#create sij by subtracting delayed pij_k by appropriate pij (see page 358 of notes)
#also add PM and OP noises appropriately. See page 517 and
#............................................................

s31_LFN = (p13_2.copy()[beg_ind:end_ind:]-p31.copy()[beg_ind:end_ind:])[::down_sample_factor]
s21_LFN = (p12_3p.copy()[beg_ind:end_ind:]-p21.copy()[beg_ind:end_ind:])[::down_sample_factor]
s32_LFN = (p23_1p.copy()[beg_ind:end_ind:]-p32.copy()[beg_ind:end_ind:])[::down_sample_factor]
s12_LFN = (p21_3.copy()[beg_ind:end_ind:]-p12.copy()[beg_ind:end_ind:])[::down_sample_factor]
s23_LFN = (p32_1.copy()[beg_ind:end_ind:]- p23.copy()[beg_ind:end_ind:])[::down_sample_factor]
s13_LFN = (p31_2p.copy()[beg_ind:end_ind:]-p13.copy()[beg_ind:end_ind:])[::down_sample_factor]

tau31_LFN = (p21.copy()[beg_ind:end_ind:]-p31.copy()[beg_ind:end_ind:])[::down_sample_factor] 
tau21_LFN = (p31.copy()[beg_ind:end_ind:]-p21.copy()[beg_ind:end_ind:])[::down_sample_factor]
tau12_LFN = (p32.copy()[beg_ind:end_ind:]-p12.copy()[beg_ind:end_ind:])[::down_sample_factor] 
tau32_LFN = (p12.copy()[beg_ind:end_ind:]-p32.copy()[beg_ind:end_ind:])[::down_sample_factor] 
tau23_LFN = (p13.copy()[beg_ind:end_ind:]-p23.copy()[beg_ind:end_ind:])[::down_sample_factor] 
tau13_LFN = (p23.copy()[beg_ind:end_ind:]-p13.copy()[beg_ind:end_ind:])[::down_sample_factor] 

eps31_LFN = tau31_LFN
eps21_LFN = tau21_LFN
eps12_LFN = tau12_LFN
eps32_LFN = tau32_LFN
eps23_LFN = tau23_LFN
eps13_LFN = tau13_LFN

#............................................................
#create epsilon_ij by just subtracting
#............................................................

del p13
del p12
del p21
del p23
del p31
del p32

del p13_2
del p12_3p
del p32_1
del p23_1p
del p21_3
del p31_2p



#............................................................
#save to file
#............................................................
#header='time s31 s21 s32 s12 s23 s13 tau31 tau21 tau12 tau32 tau23 tau13'
np.savetxt('comparison_data_fs_1e3_N=101_LFN_only.dat',np.c_[s31_LFN, s21_LFN,s32_LFN, s12_LFN, s23_LFN, s13_LFN, tau31_LFN, tau21_LFN, tau12_LFN, tau32_LFN, tau23_LFN, tau13_LFN, eps31_LFN, eps21_LFN, eps12_LFN, eps32_LFN, eps23_LFN, eps13_LFN],header='s31_LFN s21_LFN s32_LFN s12_LFN s23_LFN s13_LFN tau31_LFN tau21_LFN tau12_LFN tau32_LFN tau23_LFN tau13_LFN eps31_LFN eps21_LFN eps12_LFN eps32_LFN eps23_LFN eps13_LFN')

