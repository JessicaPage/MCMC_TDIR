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

#import argparse

def next_two_power(n):

	return int(pow(2, ceil(log(n, 2))))

def _centered_from_scipy(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

#...........................CREATING FILTERS.......................................
def filters_lagrange(delay):

	D = delay*f_samp
	i, d_frac = divmod(D,1)

	if d_frac >= 0.5:
		i+=1
		d_frac = -1*(1-d_frac)

	# delayed filters we're convolving with

	lagrange_filter = np.zeros(len(h_points))
	h_points_filter = h_points - int(i) 
	indices = np.where(np.logical_and(h_points_filter>= int(-(number_n-1)/2.0),h_points_filter <= int((number_n-1)/2.0)))

	lagrange_filter[indices[0]] = [lagrange(int(n),number_n,d_frac)*np.sinc(int(n)-d_frac) for n in h_points_filter[indices[0]]]

	return lagrange_filter


#window for LaGrange function WITH MPMATH
def lagrange(n,N,D_here):

	D=D_here
	N = float(N)
	n = float(n) 
	t_D = 0.5*(N-1)+D
	'''
	if D < 1.e-12:
		window = 1.
	else:
		window = math.pi*N/(math.sin(math.pi*t_D))*binomial(t_D,N)*binomial((N-1),(n+(N-1)*0.5))
	'''
	window = math.pi*N/(math.sin(math.pi*t_D))*binomial(t_D,N)*binomial((N-1),(n+(N-1)*0.5))
	
	return window

def S_y_proof_mass_new_frac_freq(f):
	pm_here =  np.power(3.0e-15,2)*(1+np.power(4.0e-4/f,2))*(1+np.power(f/8.0e-3,4))
	#*np.power(2*np.pi*f,-4)
	return pm_here*np.power(2*np.pi*f*const.c.value,-2)

def S_y_OMS_frac_freq(f):
	op_here =  np.power(1.5e-11,2)*np.power(2*np.pi*f/const.c.value,2)*(1+np.power(2.0e-3/f,4))
	return op_here



#cutting off bad data due to delay times and filter length, 
#indices chosen once at beginning instead of every iteration

def cut_data(D_3,D_2,D_1,D_3_p,D_2_p,D_1_p,f_rate,length):

	D_2_val = int(round(f_rate*D_2))
	D_3_val = int(round(f_rate*D_3))
	D_1_val = int(round(f_rate*D_1))

	D_2_p_val = int(round(f_rate*D_2_p))
	D_3_p_val = int(round(f_rate*D_3_p))
	D_1_p_val = int(round(f_rate*D_1_p))

	filter_cut = int(round((number_n-1)))
	#filter_cut = int(round((61-1)))


	#beg_ind = half_extra + filter_cut + D_3_val+D_2_val+D_1_val+D_3_p_val+D_2_p_val+D_1_p_val
	#beg_ind = half_extra + filter_cut 
	beg_ind = filter_cut + D_3_val+D_2_val+D_1_val+D_3_p_val+D_2_p_val+D_1_p_val

	#beg_ind = filter_cut
	#end_ind = int(two_power - filter_cut - half_extra - 1)
	end_ind = int(length - filter_cut - 1)


	return beg_ind, end_ind


def x_combo(L_3_here, L_2_here,L_3_p_here,L_2_p_here):


	lagrange_2= filters_lagrange(L_2_here)
	l_2_f = np.fft.rfft(lagrange_2,two_power)
	s_13_2_f = s13_f_convolve*l_2_f


	lagrange_2_p_2 = filters_lagrange(L_2_p_here+L_2_here)
	l_2_p_2_f = np.fft.rfft(lagrange_2_p_2,two_power)
	s_21_2p_2_f = s21_f_convolve*l_2_p_2_f

	lagrange_3_p_2_p_2 = filters_lagrange(L_3_p_here+L_2_p_here+L_2_here)
	l_3_p_2_p_2_f = np.fft.rfft(lagrange_3_p_2_p_2,two_power)
	s_12_3p_2p_2_f = s12_f_convolve*l_3_p_2_p_2_f

	lagrange_3p= filters_lagrange(L_3_p_here)
	l_3_p_f = np.fft.rfft(lagrange_3p,two_power)
	s_12_3p_f = s12_f_convolve*l_3_p_f

	lagrange_3_3p= filters_lagrange(L_3_here+L_3_p_here)
	l_3_3_p_f = np.fft.rfft(lagrange_3_3p,two_power)
	s_31_3_3p_f = s31_f_convolve*l_3_3_p_f

	lagrange_2_3_3_p = filters_lagrange(L_2_here+L_3_here+L_3_p_here)
	l_2_3_3_p_f = np.fft.rfft(lagrange_2_3_3_p,two_power)
	s_13_2_3_3p_f = s13_f_convolve*l_2_3_3_p_f

	lagrange_2_p_2_3_3_p = filters_lagrange(L_2_p_here+L_2_here+L_3_here+L_3_p_here)
	l_2_p_2_3_3_p_f = np.fft.rfft(lagrange_2_p_2_3_3_p,two_power)
	tau_21_2_p_2_3_3p_f = tau21_f_convolve*l_2_p_2_3_3_p_f 
	tau_31_2_p_2_3_3p_f = tau31_f_convolve*l_2_p_2_3_3_p_f 

	tau21_3_3p_f = tau21_f_convolve*l_3_3_p_f
	tau31_3_3p_f = tau31_f_convolve*l_3_3_p_f

	tau21_2p_2_f = tau21_f_convolve*l_2_p_2_f
	tau31_2p_2_f = tau31_f_convolve*l_2_p_2_f

	eps_31_2_p_2_3_3p_f = eps31_f_convolve*l_2_p_2_3_3_p_f 

	eps31_3_3p_f = eps31_f_convolve*l_3_3_p_f

	eps31_2p_2_f = eps31_f_convolve*l_2_p_2_f

	eps_21_2_p_2_3_3p_f = eps21_f_convolve*l_2_p_2_3_3_p_f 

	eps21_3_3p_f = eps21_f_convolve*l_3_3_p_f

	eps21_2p_2_f = eps21_f_convolve*l_2_p_2_f

	eps12_3p_f = eps12_f_convolve*l_3_p_f
	tau12_3p_f = tau12_f_convolve*l_3_p_f

	eps12_3p_2p_2_f = eps12_f_convolve*l_3_p_2_p_2_f
	tau12_3p_2p_2_f = tau12_f_convolve*l_3_p_2_p_2_f

	eps13_2_f = eps13_f_convolve*l_2_f
	tau13_2_f = tau13_f_convolve*l_2_f

	eps13_2_3_3p_f = eps13_f_convolve*l_2_3_3_p_f
	tau13_2_3_3p_f = tau13_f_convolve*l_2_3_3_p_f

	x_combo_f_domain = s31_f_subtract + s_13_2_f + s_21_2p_2_f + s_12_3p_2p_2_f - s21_f_subtract - s_12_3p_f - s_31_3_3p_f - s_13_2_3_3p_f + 0.5*(tau_21_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f - tau21_3_3p_f + tau31_3_3p_f - tau21_2p_2_f + tau31_2p_2_f + tau21_f_subtract - tau31_f_subtract) + 0.5*(eps_31_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f + eps31_3_3p_f - tau31_3_3p_f - eps31_2p_2_f + tau31_2p_2_f - eps31_f_subtract + tau31_f_subtract) - 0.5*(eps_21_2_p_2_3_3p_f - tau_21_2_p_2_3_3p_f - eps21_3_3p_f + tau21_3_3p_f + eps21_2p_2_f - tau21_2p_2_f - eps21_f_subtract + tau21_f_subtract) + eps12_3p_f - tau12_3p_f - eps12_3p_2p_2_f + tau12_3p_2p_2_f - eps13_2_f + tau13_2_f + eps13_2_3_3p_f - tau13_2_3_3p_f
	#x_combo_f_domain = s31_f_convolve + s_13_2_f + s_21_2p_2_f + s_12_3p_2p_2_f - s21_f_convolve - s_12_3p_f - s_31_3_3p_f - s_13_2_3_3p_f + 0.5*(tau_21_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f - tau21_3_3p_f + tau31_3_3p_f - tau21_2p_2_f + tau31_2p_2_f + tau21_f_convolve - tau31_f_convolve) + 0.5*(eps_31_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f + eps31_3_3p_f - tau31_3_3p_f - eps31_2p_2_f + tau31_2p_2_f - eps31_f_convolve + tau31_f_convolve) - 0.5*(eps_21_2_p_2_3_3p_f - tau_21_2_p_2_3_3p_f - eps21_3_3p_f + tau21_3_3p_f + eps21_2p_2_f - tau21_2p_2_f - eps21_f_convolve + tau21_f_convolve) + eps12_3p_f - tau12_3p_f - eps12_3p_2p_2_f + tau12_3p_2p_2_f - eps13_2_f + tau13_2_f + eps13_2_3_3p_f - tau13_2_3_3p_f

	x_combo_val = np.fft.irfft(x_combo_f_domain, norm='ortho')

	centered_x = _centered_from_scipy(x_combo_val,length)

	x_combo_val = centered_x[beg_ind:end_ind]

	x_combo_f_domain = np.fft.rfft(window*x_combo_val,norm='ortho')[indices_f_band]

	return x_combo_f_domain


def y_combo(L_3_here, L_1_here,L_3_p_here,L_1_p_here):


	lagrange_3= filters_lagrange(L_3_here)
	l_3_f = np.fft.rfft(lagrange_3,two_power)
	s_21_3_f = s21_f_convolve*l_3_f


	lagrange_3_p_3 = filters_lagrange(L_3_p_here+L_3_here)
	l_3_p_3_f = np.fft.rfft(lagrange_3_p_3,two_power)
	s_32_3p_3_f = s32_f_convolve*l_3_p_3_f

	lagrange_1_p_3_p_3 = filters_lagrange(L_1_p_here+L_3_p_here+L_3_here)
	l_1_p_3_p_3_f = np.fft.rfft(lagrange_1_p_3_p_3,two_power)
	s_23_1p_3p_3_f = s23_f_convolve*l_1_p_3_p_3_f

	lagrange_1p= filters_lagrange(L_1_p_here)
	l_1_p_f = np.fft.rfft(lagrange_1p,two_power)
	s_23_1p_f = s23_f_convolve*l_1_p_f

	lagrange_1_1p= filters_lagrange(L_1_here+L_1_p_here)
	l_1_1_p_f = np.fft.rfft(lagrange_1_1p,two_power)
	s_12_1_1p_f = s12_f_convolve*l_1_1_p_f

	lagrange_3_1_1_p = filters_lagrange(L_3_here+L_1_here+L_1_p_here)
	l_3_1_1_p_f = np.fft.rfft(lagrange_3_1_1_p,two_power)
	s_21_3_1_1p_f = s21_f_convolve*l_3_1_1_p_f

	lagrange_3_p_3_1_1_p = filters_lagrange(L_3_p_here+L_3_here+L_1_here+L_1_p_here)
	l_3_p_3_1_1_p_f = np.fft.rfft(lagrange_3_p_3_1_1_p,two_power)
	tau_32_3_p_3_1_1p_f = tau32_f_convolve*l_3_p_3_1_1_p_f 
	tau_12_3_p_3_1_1p_f = tau12_f_convolve*l_3_p_3_1_1_p_f 

	tau32_1_1p_f = tau32_f_convolve*l_1_1_p_f
	tau12_1_1p_f = tau12_f_convolve*l_1_1_p_f

	tau32_3p_3_f = tau32_f_convolve*l_3_p_3_f
	tau12_3p_3_f = tau12_f_convolve*l_3_p_3_f

	eps_12_3_p_3_1_1p_f = eps12_f_convolve*l_3_p_3_1_1_p_f 

	eps12_1_1p_f = eps12_f_convolve*l_1_1_p_f

	eps12_3p_3_f = eps12_f_convolve*l_3_p_3_f

	eps_32_3_p_3_1_1p_f = eps32_f_convolve*l_3_p_3_1_1_p_f 

	eps32_1_1p_f = eps32_f_convolve*l_1_1_p_f

	eps32_3p_3_f = eps32_f_convolve*l_3_p_3_f

	eps23_1p_f = eps23_f_convolve*l_1_p_f
	tau23_1p_f = tau23_f_convolve*l_1_p_f

	eps23_1p_3p_3_f = eps23_f_convolve*l_1_p_3_p_3_f
	tau23_1p_3p_3_f = tau23_f_convolve*l_1_p_3_p_3_f

	eps21_3_f = eps21_f_convolve*l_3_f
	tau21_3_f = tau21_f_convolve*l_3_f

	eps21_3_1_1p_f = eps21_f_convolve*l_3_1_1_p_f
	tau21_3_1_1p_f = tau21_f_convolve*l_3_1_1_p_f

	y_combo_f_domain = s12_f_subtract + s_21_3_f + s_32_3p_3_f + s_23_1p_3p_3_f - s32_f_subtract - s_23_1p_f - s_12_1_1p_f - s_21_3_1_1p_f + 0.5*(tau_32_3_p_3_1_1p_f - tau_12_3_p_3_1_1p_f - tau32_1_1p_f + tau12_1_1p_f - tau32_3p_3_f + tau12_3p_3_f + tau32_f_subtract - tau12_f_subtract) + 0.5*(eps_12_3_p_3_1_1p_f - tau_12_3_p_3_1_1p_f + eps12_1_1p_f - tau12_1_1p_f - eps12_3p_3_f + tau12_3p_3_f - eps12_f_subtract + tau12_f_subtract) - 0.5*(eps_32_3_p_3_1_1p_f - tau_32_3_p_3_1_1p_f - eps32_1_1p_f + tau32_1_1p_f + eps32_3p_3_f - tau32_3p_3_f - eps32_f_subtract + tau32_f_subtract) + eps23_1p_f - tau23_1p_f - eps23_1p_3p_3_f + tau23_1p_3p_3_f - eps21_3_f + tau21_3_f + eps21_3_1_1p_f - tau21_3_1_1p_f
	#y_combo_f_domain = s31_f_convolve + s_13_2_f + s_21_2p_2_f + s_12_3p_2p_2_f - s21_f_convolve - s_12_3p_f - s_31_3_3p_f - s_13_2_3_3p_f + 0.5*(tau_21_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f - tau21_3_3p_f + tau31_3_3p_f - tau21_2p_2_f + tau31_2p_2_f + tau21_f_convolve - tau31_f_convolve) + 0.5*(eps_31_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f + eps31_3_3p_f - tau31_3_3p_f - eps31_2p_2_f + tau31_2p_2_f - eps31_f_convolve + tau31_f_convolve) - 0.5*(eps_21_2_p_2_3_3p_f - tau_21_2_p_2_3_3p_f - eps21_3_3p_f + tau21_3_3p_f + eps21_2p_2_f - tau21_2p_2_f - eps21_f_convolve + tau21_f_convolve) + eps12_3p_f - tau12_3p_f - eps12_3p_2p_2_f + tau12_3p_2p_2_f - eps13_2_f + tau13_2_f + eps13_2_3_3p_f - tau13_2_3_3p_f

	y_combo_val = np.fft.irfft(y_combo_f_domain, norm='ortho')

	centered_y = _centered_from_scipy(y_combo_val,length)

	y_combo_val = centered_y[beg_ind:end_ind]


	y_combo_f_domain = np.fft.rfft(window*y_combo_val,norm='ortho')[indices_f_band]
	

	return y_combo_f_domain

def z_combo(L_2_here, L_1_here,L_2_p_here,L_1_p_here):


	lagrange_1= filters_lagrange(L_1_here)
	l_1_f = np.fft.rfft(lagrange_1,two_power)
	s_32_1_f = s32_f_convolve*l_1_f


	lagrange_1_p_1 = filters_lagrange(L_1_p_here+L_1_here)
	l_1_p_1_f = np.fft.rfft(lagrange_1_p_1,two_power)
	s_13_1p_1_f = s13_f_convolve*l_1_p_1_f

	lagrange_2_p_1_p_1 = filters_lagrange(L_2_p_here+L_1_p_here+L_1_here)
	l_2_p_1_p_1_f = np.fft.rfft(lagrange_2_p_1_p_1,two_power)
	s_31_2p_1p_1_f = s31_f_convolve*l_2_p_1_p_1_f

	lagrange_2p= filters_lagrange(L_2_p_here)
	l_2_p_f = np.fft.rfft(lagrange_2p,two_power)
	s_31_2p_f = s31_f_convolve*l_2_p_f

	lagrange_2_2p= filters_lagrange(L_2_here+L_2_p_here)
	l_2_2_p_f = np.fft.rfft(lagrange_2_2p,two_power)
	s_23_2_2p_f = s23_f_convolve*l_2_2_p_f

	lagrange_1_2_2_p = filters_lagrange(L_1_here+L_2_here+L_2_p_here)
	l_1_2_2_p_f = np.fft.rfft(lagrange_1_2_2_p,two_power)
	s_32_1_2_2p_f = s32_f_convolve*l_1_2_2_p_f

	lagrange_1_p_1_2_2_p = filters_lagrange(L_1_p_here+L_1_here+L_2_here+L_2_p_here)
	l_1_p_1_2_2_p_f = np.fft.rfft(lagrange_1_p_1_2_2_p,two_power)
	tau_13_1_p_1_2_2p_f = tau13_f_convolve*l_1_p_1_2_2_p_f 
	tau_23_1_p_1_2_2p_f = tau23_f_convolve*l_1_p_1_2_2_p_f 

	tau13_2_2p_f = tau13_f_convolve*l_2_2_p_f
	tau23_2_2p_f = tau23_f_convolve*l_2_2_p_f

	tau13_1p_1_f = tau13_f_convolve*l_1_p_1_f
	tau23_1p_1_f = tau23_f_convolve*l_1_p_1_f

	eps_23_1_p_1_2_2p_f = eps23_f_convolve*l_1_p_1_2_2_p_f 

	eps23_2_2p_f = eps23_f_convolve*l_2_2_p_f

	eps23_1p_1_f = eps23_f_convolve*l_1_p_1_f

	eps_13_1_p_1_2_2p_f = eps13_f_convolve*l_1_p_1_2_2_p_f 

	eps13_2_2p_f = eps13_f_convolve*l_2_2_p_f

	eps13_1p_1_f = eps13_f_convolve*l_1_p_1_f

	eps31_2p_f = eps31_f_convolve*l_2_p_f
	tau31_2p_f = tau31_f_convolve*l_2_p_f

	eps31_2p_1p_1_f = eps31_f_convolve*l_2_p_1_p_1_f
	tau31_2p_1p_1_f = tau31_f_convolve*l_2_p_1_p_1_f

	eps32_1_f = eps32_f_convolve*l_1_f
	tau32_1_f = tau32_f_convolve*l_1_f

	eps32_1_2_2p_f = eps32_f_convolve*l_1_2_2_p_f
	tau32_1_2_2p_f = tau32_f_convolve*l_1_2_2_p_f

	z_combo_f_domain = s23_f_subtract + s_32_1_f + s_13_1p_1_f + s_31_2p_1p_1_f - s13_f_subtract - s_31_2p_f - s_23_2_2p_f - s_32_1_2_2p_f + 0.5*(tau_13_1_p_1_2_2p_f - tau_23_1_p_1_2_2p_f - tau13_2_2p_f + tau23_2_2p_f - tau13_1p_1_f + tau23_1p_1_f + tau13_f_subtract - tau23_f_subtract) + 0.5*(eps_23_1_p_1_2_2p_f - tau_23_1_p_1_2_2p_f + eps23_2_2p_f - tau23_2_2p_f - eps23_1p_1_f + tau23_1p_1_f - eps23_f_subtract + tau23_f_subtract) - 0.5*(eps_13_1_p_1_2_2p_f - tau_13_1_p_1_2_2p_f - eps13_2_2p_f + tau13_2_2p_f + eps13_1p_1_f - tau13_1p_1_f - eps13_f_subtract + tau13_f_subtract) + eps31_2p_f - tau31_2p_f - eps31_2p_1p_1_f + tau31_2p_1p_1_f - eps32_1_f + tau32_1_f + eps32_1_2_2p_f - tau32_1_2_2p_f
	#z_combo_f_domain = s31_f_convolve + s_13_2_f + s_21_2p_2_f + s_12_3p_2p_2_f - s21_f_convolve - s_12_3p_f - s_31_3_3p_f - s_13_2_3_3p_f + 0.5*(tau_21_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f - tau21_3_3p_f + tau31_3_3p_f - tau21_2p_2_f + tau31_2p_2_f + tau21_f_convolve - tau31_f_convolve) + 0.5*(eps_31_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f + eps31_3_3p_f - tau31_3_3p_f - eps31_2p_2_f + tau31_2p_2_f - eps31_f_convolve + tau31_f_convolve) - 0.5*(eps_21_2_p_2_3_3p_f - tau_21_2_p_2_3_3p_f - eps21_3_3p_f + tau21_3_3p_f + eps21_2p_2_f - tau21_2p_2_f - eps21_f_convolve + tau21_f_convolve) + eps12_3p_f - tau12_3p_f - eps12_3p_2p_2_f + tau12_3p_2p_2_f - eps13_2_f + tau13_2_f + eps13_2_3_3p_f - tau13_2_3_3p_f


	z_combo_val = np.fft.irfft(z_combo_f_domain, norm='ortho')



	centered_z = _centered_from_scipy(z_combo_val,length)


	z_combo_val = centered_z[beg_ind:end_ind]


	z_combo_f_domain = np.fft.rfft(window*z_combo_val,norm='ortho')[indices_f_band]

	return z_combo_f_domain
def covariance_equal_arm():


	a = 16*np.power(np.sin(2*np.pi*f_band*avg_L),2)*Sy_OP+(8*np.power(np.sin(4*np.pi*f_band*avg_L),2)+32*np.power(np.sin(2*np.pi*f_band*avg_L),2))*Sy_PM

	b_ = -4*np.sin(2*np.pi*f_band*avg_L)*np.sin(4*np.pi*f_band*avg_L)*(4*Sy_PM+Sy_OP)


	return 2*a,2*b_

#See pages 609-613
def likelihood_analytical_equal_arm(x_combo_f,y_combo_f,z_combo_f):

	x_x = np.real(x_combo_f)
	x_y = np.imag(x_combo_f)

	y_x = np.real(y_combo_f)
	y_y = np.imag(y_combo_f)

	z_x = np.real(z_combo_f)
	z_y = np.imag(z_combo_f)

	chi_2 = 1/determinant*(A_*(x_x**2+x_y**2+y_x**2+y_y**2+z_x**2+z_y**2) + 2*B_*(x_x*y_x+x_y*y_y+x_x*z_x+x_y*z_y+y_x*z_x+y_y*z_y))

	value = -1*np.sum(chi_2) - log_term_factor - np.sum(log_term_determinant) 

	print('logL')
	print(value)

	plt.loglog(np.fft.rfftfreq(length,1/f_s),np.abs(np.fft.rfft(s31,norm='ortho')),label='s31')
	plt.loglog(f_band,np.abs(x_combo_f)**2,label='TDI 1.5 X Channel Residual')
	plt.loglog(f_band,a,label=r'$\Sigma_{00}$')
	plt.xlim(1e-4,0.1)
	plt.legend()
	plt.title('Non-Moving Arms, Doppler, Ranging and Clock Noises Disabled')
	plt.savefig('Non-Moving_Arms_Doppler_and_Ranging_and_Clock_disabled.png')
	plt.show()

	return value,np.sum(log_term_determinant),np.sum(chi_2)	

	
#........................................................................................
#.............................. Raw  PM DATA   ..........................................
#........................................................................................
#Fixed time delays (c = 1)

'''
#simulated delay times in seconds rounded to 9 decimal places
L_1 = 8.339095192
L_1_p = 8.338994879
L_2 = 8.338867041
L_2_p = 8.339095192
L_3 = 8.338994879
L_3_p = 8.338867041


#simulated delay times in seconds rounded to 9 decimal places
L_1 = 8.339095
L_1_p =8.33899493
L_2 = 8.33886739
L_2_p = 8.33909552
L_3 = 8.33899451
L_3_p = 8.33886696
'''

L_1 = 8.3044
L_1_p = 8.3028
L_2 = 8.3315
L_2_p = 8.3324
L_3 = 8.3315
L_3_p = 8.3324
'''
L_1_p = 8.3044
L_1 = 8.3028
L_2_p = 8.3315
L_2 = 8.3324
L_3_p = 8.3315
L_3 = 8.3324
'''
f_s = 4

f_samp = 4


number_n = 29

f_min = 1.0e-4 # (= 0.0009765625)
f_max =1.0e-1



#............................................................
#................................... DATA . ....................................
#............................................................
#data =  np.genfromtxt('./../Data_Simulation/data_fs_4_N=49_FOR_RUN.dat',names=True)
data =  np.genfromtxt('LISA_Instrument_RR_more_noise.dat',names=True)
#data =  np.genfromtxt('./../Data_Simulation/data_fs_4_N=49.dat',names=True)
#data =  np.genfromtxt('./../Data_Simulation/data_fs_4_N=49.dat',names=True)


s31 = data['s31']
if len(s31)%2!=0:
	s31 = data['s31'][:-1]
	s21 = data['s21'][:-1]
	s32 = data['s32'][:-1]
	s12 = data['s12'][:-1]
	s23 = data['s23'][:-1]
	s13 = data['s13'][:-1]


	tau31 = data['tau31'][:-1]
	tau21 = data['tau21'][:-1]
	tau12 = data['tau12'][:-1]
	tau32 = data['tau32'][:-1]
	tau23 = data['tau23'][:-1]
	tau13 = data['tau13'][:-1]

	eps31 = data['eps31'][:-1]
	eps21 = data['eps21'][:-1]
	eps12 = data['eps12'][:-1]
	eps32 = data['eps32'][:-1]
	eps23 = data['eps23'][:-1]
	eps13 = data['eps13'][:-1]
else:
	s31 = data['s31']
	s21 = data['s21']
	s32 = data['s32']
	s12 = data['s12']
	s23 = data['s23']
	s13 = data['s13']


	tau31 = data['tau31']
	tau21 = data['tau21']
	tau12 = data['tau12']
	tau32 = data['tau32']
	tau23 = data['tau23']
	tau13 = data['tau13']

	eps31 = data['eps31']
	eps21 = data['eps21']
	eps12 = data['eps12']
	eps32 = data['eps32']
	eps23 = data['eps23']
	eps13 = data['eps13']



length = len(s31)

del data
'''
two_power = next_two_power(length)
m = two_power-length+1

'''
m =301
#avoid circular convolution
two_power=length+m-1

two_power = next_fast_len(two_power,real=True)
#two_power = next_two_power(two_power)

m = two_power-length+1

print('length')
print(length)
print('two_power')
print(two_power)
print('m')
print(m)

#FFT's for manual FFT convolution in interpolation step
s31_f_convolve = np.fft.rfft(s31,two_power,norm='ortho')
s21_f_convolve = np.fft.rfft(s21,two_power,norm='ortho')
s32_f_convolve = np.fft.rfft(s32,two_power,norm='ortho')
s12_f_convolve = np.fft.rfft(s12,two_power,norm='ortho')
s23_f_convolve = np.fft.rfft(s23,two_power,norm='ortho')
s13_f_convolve = np.fft.rfft(s13,two_power,norm='ortho')
tau31_f_convolve = np.fft.rfft(tau31,two_power,norm='ortho')
tau21_f_convolve = np.fft.rfft(tau21,two_power,norm='ortho')
tau32_f_convolve = np.fft.rfft(tau32,two_power,norm='ortho')
tau12_f_convolve = np.fft.rfft(tau12,two_power,norm='ortho')
tau23_f_convolve = np.fft.rfft(tau23,two_power,norm='ortho')
tau13_f_convolve = np.fft.rfft(tau13,two_power,norm='ortho')
eps31_f_convolve = np.fft.rfft(eps31,two_power,norm='ortho')
eps21_f_convolve = np.fft.rfft(eps21,two_power,norm='ortho')
eps32_f_convolve = np.fft.rfft(eps32,two_power,norm='ortho')
eps12_f_convolve = np.fft.rfft(eps12,two_power,norm='ortho')
eps23_f_convolve = np.fft.rfft(eps23,two_power,norm='ortho')
eps13_f_convolve = np.fft.rfft(eps13,two_power,norm='ortho')

extra_pad = two_power-length
half_extra = (extra_pad)//2

#FFTs for TDI subtraction (Zeroes have to be padded on either side instead of at end.)
s31_half_pad = np.pad(s31,(half_extra,half_extra),'constant')
s21_half_pad = np.pad(s21,(half_extra,half_extra),'constant')
s32_half_pad = np.pad(s32,(half_extra,half_extra),'constant')
s12_half_pad = np.pad(s12,(half_extra,half_extra),'constant')
s23_half_pad = np.pad(s23,(half_extra,half_extra),'constant')
s13_half_pad = np.pad(s13,(half_extra,half_extra),'constant')
tau31_half_pad = np.pad(tau31,(half_extra,half_extra),'constant')
tau21_half_pad = np.pad(tau21,(half_extra,half_extra),'constant')
tau32_half_pad = np.pad(tau32,(half_extra,half_extra),'constant')
tau12_half_pad = np.pad(tau12,(half_extra,half_extra),'constant')
tau23_half_pad = np.pad(tau23,(half_extra,half_extra),'constant')
tau13_half_pad = np.pad(tau13,(half_extra,half_extra),'constant')
eps31_half_pad = np.pad(eps31,(half_extra,half_extra),'constant')
eps21_half_pad = np.pad(eps21,(half_extra,half_extra),'constant')
eps32_half_pad = np.pad(eps32,(half_extra,half_extra),'constant')
eps12_half_pad = np.pad(eps12,(half_extra,half_extra),'constant')
eps23_half_pad = np.pad(eps23,(half_extra,half_extra),'constant')
eps13_half_pad = np.pad(eps13,(half_extra,half_extra),'constant')

s31_f_subtract = np.fft.rfft(s31_half_pad,norm='ortho')
s21_f_subtract = np.fft.rfft(s21_half_pad,norm='ortho')
s32_f_subtract = np.fft.rfft(s32_half_pad,norm='ortho')
s12_f_subtract = np.fft.rfft(s12_half_pad,norm='ortho')
s23_f_subtract = np.fft.rfft(s23_half_pad,norm='ortho')
s13_f_subtract = np.fft.rfft(s13_half_pad,norm='ortho')
tau31_f_subtract = np.fft.rfft(tau31_half_pad,norm='ortho')
tau21_f_subtract = np.fft.rfft(tau21_half_pad,norm='ortho')
tau32_f_subtract = np.fft.rfft(tau32_half_pad,norm='ortho')
tau12_f_subtract = np.fft.rfft(tau12_half_pad,norm='ortho')
tau23_f_subtract = np.fft.rfft(tau23_half_pad,norm='ortho')
tau13_f_subtract = np.fft.rfft(tau13_half_pad,norm='ortho')
eps31_f_subtract = np.fft.rfft(eps31_half_pad,norm='ortho')
eps21_f_subtract = np.fft.rfft(eps21_half_pad,norm='ortho')
eps32_f_subtract = np.fft.rfft(eps32_half_pad,norm='ortho')
eps12_f_subtract = np.fft.rfft(eps12_half_pad,norm='ortho')
eps23_f_subtract = np.fft.rfft(eps23_half_pad,norm='ortho')
eps13_f_subtract = np.fft.rfft(eps13_half_pad,norm='ortho')




nearest_number = m


# number points in filter
if nearest_number%2 == 0:
	h_points = np.arange(-nearest_number/2.0,nearest_number/2.0,1)
else:
	h_points = np.arange(-(nearest_number-1)/2.0,(nearest_number-1)/2.0+1,1)



#min and max values from orbit file
low = 8.338489422296977
high = 8.339102379118449


L_arm = 2.5e9
avg_L = L_arm/const.c.value

beg_ind,end_ind = cut_data(L_3,L_2,L_1,L_3_p,L_2_p,L_1_p,f_s,length)
window = cosine(length)[beg_ind:end_ind:]
f_band = np.fft.rfftfreq(len(window),1/f_s)
indices_f_band = np.where(np.logical_and(f_band>=f_min, f_band<=f_max))
f_band=f_band[indices_f_band]

f_band_before_cut = np.fft.rfftfreq(two_power,1/f_s)

#new noise PSDs afetr beginning split int implementation FRACTIONAL FREQUENCY PSD
Sy_PM = S_y_proof_mass_new_frac_freq(f_band)
Sy_OP = S_y_OMS_frac_freq(f_band)

a,b_ = covariance_equal_arm()

#Needed in inverse calculation
A_ = a**2 - b_**2
B_ = b_**2 - a*b_

#det_here = np.array(np.linalg.det(np.pi*covariance_here),dtype=np.complex128)
log_term_factor = 3*np.log(np.pi)
determinant = a*A_+2*b_*B_
log_term_determinant = np.log(determinant)


initial_L_1 = L_1
initial_L_1_p = L_1_p
initial_L_2 = L_2
initial_L_3 = L_3
initial_L_2_p = L_2_p
initial_L_3_p = L_3_p

'''
initial_L_1 = np.random.uniform(low,high)
initial_L_1_p = np.random.uniform(low,high)
initial_L_2 = np.random.uniform(low,high)
initial_L_2_p = np.random.uniform(low,high)
initial_L_3 = np.random.uniform(low,high)
initial_L_3_p = np.random.uniform(low,high)
'''
#initial delays accepted into the chain
accept = 1
x_combo_initial = x_combo(initial_L_3,initial_L_2,initial_L_3_p, initial_L_2_p)
y_combo_initial = y_combo(initial_L_3,initial_L_1,initial_L_3_p,initial_L_1_p)
z_combo_initial = z_combo(initial_L_2,initial_L_1,initial_L_2_p,initial_L_1_p)

old_likelihood,determ_here,chi_2_here = likelihood_analytical_equal_arm(x_combo_initial,y_combo_initial,z_combo_initial)


sys.exit()
