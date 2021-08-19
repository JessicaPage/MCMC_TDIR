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
#import argparse

def next_two_power(n):

	return int(pow(2, ceil(log(n, 2))))


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

	#filter_cut = int(round((number_n-1)))
	filter_cut = int(round((61-1)))


	beg_ind = filter_cut+D_3_val+D_2_val+D_1_val+D_3_p_val+D_2_p_val+D_1_p_val
	#beg_ind = filter_cut
	end_ind = int(length-filter_cut-1)


	return beg_ind, end_ind
'''
def x_combo(L_3_here, L_2_here,L_3_p_here,L_2_p_here):


	lagrange_s_13_2= filters_lagrange(L_2_here)
	s_13_2 = fftconvolve(s13,lagrange_s_13_2,'same')


	lagrange_s_12_3p= filters_lagrange(L_3_p_here)
	s_12_3p = fftconvolve(s12,lagrange_s_12_3p,'same')


	G = s21+s_12_3p
	F = s31+s_13_2
	H = tau21-tau31

	lagrange_A_2p_2 = filters_lagrange((L_2_p_here+L_2_here))
	A = fftconvolve(G,lagrange_A_2p_2,'same')

	lagrange_B_3_3p = filters_lagrange((L_3_here+L_3_p_here))
	B = fftconvolve(F,lagrange_B_3_3p,'same')

	lagrange_C_2p_2_3_3p = filters_lagrange((L_2_p_here+L_2_here+L_3_here+L_3_p_here))
	C = fftconvolve(H,lagrange_C_2p_2_3_3p,'same')

	D_equation = fftconvolve(H,lagrange_B_3_3p,'same')

	E = fftconvolve(H,lagrange_A_2p_2,'same')


	J = eps31-tau31
	J_three = fftconvolve(J,lagrange_B_3_3p,'same')
	J_two = fftconvolve(J,lagrange_A_2p_2,'same')
	J_all = fftconvolve(J,lagrange_C_2p_2_3_3p,'same')

	P = eps21-tau21
	P_three = fftconvolve(P,lagrange_B_3_3p,'same')
	P_two = fftconvolve(P,lagrange_A_2p_2,'same')
	P_all = fftconvolve(P,lagrange_C_2p_2_3_3p,'same')

	Q = eps12-tau12
	Q_one = fftconvolve(Q,lagrange_s_12_3p,'same')
	Q_three = fftconvolve(Q_one,lagrange_A_2p_2,'same')

	R = eps13-tau13
	R_one = fftconvolve(R,lagrange_s_13_2,'same')
	R_three = fftconvolve(R_one,lagrange_B_3_3p,'same')


	#TDI X combo RR X_1.5 
	x_combo_val = (F+A)-(G+B)+0.5*(C-D_equation-E+H) +0.5*(J_all+J_three-J_two-J) -0.5*(P_all-P_three+P_two-P)+Q_one-Q_three-R_one+R_three
	plt.semilogy(np.abs(x_combo_val))
	plt.title('time domain')
	plt.show()
	x_combo_val = x_combo_val[beg_ind:end_ind:]
	plt.semilogy(np.abs(x_combo_val))
	plt.title('time domain')
	plt.show()
	x_combo_f_domain = np.fft.rfft(window*x_combo_val,norm='ortho')[indices_f_band]
	
	plt.loglog(f_band,np.abs(x_combo_f_domain))
	plt.show()
	x_combo_val = np.fft.irfft(x_combo_f_domain, norm='ortho')

	plt.semilogy(np.abs(x_combo_val))
	plt.title('time domain')
	plt.show()

	return x_combo_f_domain
'''

'''
def x_combo(L_3_here, L_2_here,L_3_p_here,L_2_p_here):


	lagrange_s_13_2= filters_lagrange(L_2_here)
	plt.plot(lagrange_s_13_2)
	plt.title('filter time domain')
	plt.show()
	l_2_f = np.fft.rfft(lagrange_s_13_2,two_power,norm='ortho')
	plt.plot(np.abs(l_2_f))
	plt.title('filter in freq domain')
	plt.show()
	s_13_2_f = s13_f_convolve*l_2_f
	#s_13_2 = fftconvolve(s13,lagrange_s_13_2,'same')


	lagrange_s_12_3p= filters_lagrange(L_3_p_here)
	l_3_p_f = np.fft.rfft(lagrange_s_12_3p,two_power,norm='ortho')
	s_12_3p_f = s12_f_convolve*l_3_p_f
	#s_12_3p = fftconvolve(s12,lagrange_s_12_3p,'same')


	#G = s21+s_12_3p
	G = s21_f_subtract+s_12_3p_f
	#G_convolve = s21_f_convolve+s_12_3p_f
	#F = s31+s_13_2
	F = s31_f_subtract + s_13_2_f
	#F_convolve = s31_f_convolve + s_13_2_f

	#H = tau21-tau31
	H = tau21_f_subtract-tau31_f_subtract
	#H_convolve = tau21_f_convolve-tau31_f_convolve


	lagrange_A_2p_2 = filters_lagrange((L_2_p_here+L_2_here))
	l_2_2p_f = np.fft.rfft(lagrange_A_2p_2,two_power,norm='ortho')
	A = G*l_2_2p_f
	#A = G_convolve*l_2_2p_f

	#A = fftconvolve(G,lagrange_A_2p_2,'same')

	lagrange_B_3_3p = filters_lagrange((L_3_here+L_3_p_here))
	l_3_3p_f = np.fft.rfft(lagrange_B_3_3p,two_power,norm='ortho')
	B = F*l_3_3p_f
	#B = F_convolve*l_3_3p_f

	#B = fftconvolve(F,lagrange_B_3_3p,'same')

	lagrange_C_2p_2_3_3p = filters_lagrange((L_2_p_here+L_2_here+L_3_here+L_3_p_here))
	l_3_3p_2p_2_f = np.fft.rfft(lagrange_C_2p_2_3_3p,two_power,norm='ortho')
	C = H*l_3_3p_2p_2_f
	#C = H_convolve*l_3_3p_2p_2_f

	#C = fftconvolve(H,lagrange_C_2p_2_3_3p,'same')

	#D_equation = fftconvolve(H,lagrange_B_3_3p,'same')
	D_equation = H*l_3_3p_f
	#D_equation = H_convolve*l_3_3p_f

	#E = fftconvolve(H,lagrange_A_2p_2,'same')
	E = H*l_2_2p_f
	#E = H_convolve*l_2_2p_f

	#J = eps31-tau31
	J = eps31_f_subtract-tau31_f_subtract
	#J_convolve = eps31_f_convolve-tau31_f_convolve

	#J_three = fftconvolve(J,lagrange_B_3_3p,'same')
	J_three = J*l_3_3p_f
	#J_three = J_convolve*l_3_3p_f

	#J_two = fftconvolve(J,lagrange_A_2p_2,'same')
	J_two = J*l_2_2p_f
	#J_two = J_convolve*l_2_2p_f

	#J_all = fftconvolve(J,lagrange_C_2p_2_3_3p,'same')
	J_all = J*l_3_3p_2p_2_f
	#J_all = J_convolve*l_3_3p_2p_2_f

	#P = eps21-tau21
	P = eps21_f_subtract-tau21_f_subtract
	#P_convolve = eps21_f_convolve-tau21_f_convolve

	#P_three = fftconvolve(P,lagrange_B_3_3p,'same')
	P_three = P*l_3_3p_f
	#P_three = P_convolve*l_3_3p_f

	#P_two = fftconvolve(P,lagrange_A_2p_2,'same')
	P_two = P*l_2_2p_f
	#P_two = P_convolve*l_2_2p_f

	#P_all = fftconvolve(P,lagrange_C_2p_2_3_3p,'same')
	P_all = P*l_3_3p_2p_2_f
	#P_all = P_convolve*l_3_3p_2p_2_f



	#Q = eps12-tau12
	Q = eps12_f_subtract-tau12_f_subtract
	#Q_convolve = eps12_f_convolve-tau12_f_convolve

	#Q_one = fftconvolve(Q,lagrange_s_12_3p,'same')
	Q_one = Q*l_3_p_f
	#Q_one = Q_convolve*l_3_p_f

	#Q_three = fftconvolve(Q_one,lagrange_A_2p_2,'same')
	Q_three = Q_one*l_2_2p_f

	#R = eps13-tau13
	R = eps13_f_subtract-tau13_f_subtract
	#R_convolve = eps13_f_convolve-tau13_f_convolve


	#R_one = fftconvolve(R,lagrange_s_13_2,'same')
	R_one = R*l_2_f
	#R_one = R_convolve*l_2_f

	#R_three = fftconvolve(R_one,lagrange_B_3_3p,'same')
	R_three = R_one*l_3_3p_f



	#TDI X combo RR X_1.5 
	x_combo_f_domain = (F+A)-(G+B)+0.5*(C-D_equation-E+H) +0.5*(J_all+J_three-J_two-J) -0.5*(P_all-P_three+P_two-P)+Q_one-Q_three-R_one+R_three
	#x_combo_f_domain = (F_convolve+A)-(G_convolve+B)+0.5*(C-D_equation-E+H_convolve) +0.5*(J_all+J_three-J_two-J_convolve) -0.5*(P_all-P_three+P_two-P_convolve)+Q_one-Q_three-R_one+R_three

	plt.loglog(f_band,np.abs(x_combo_f_domain))
	plt.show()
	x_combo_val = np.fft.irfft(x_combo_f_domain, norm='ortho')

	plt.semilogy(np.abs(x_combo_val))
	plt.title('time domain')
	plt.show()

	#x_combo_f_domain = np.fft.rfft(window*x_combo_val,norm='ortho')[indices_f_band]
	

	return x_combo_f_domain

'''

'''
def x_combo(L_3_here, L_2_here,L_3_p_here,L_2_p_here):


	lagrange_2= filters_lagrange(L_2_here)
	l_2_f = np.fft.rfft(lagrange_2,two_power,norm='ortho')
	s_13_2_f = s13_f_convolve*l_2_f


	lagrange_2_p_2 = filters_lagrange(L_2_p_here+L_2_here)
	l_2_p_2_f = np.fft.rfft(lagrange_2_p_2,two_power,norm='ortho')
	s_21_2p_2_f = s21_f_convolve*l_2_p_2_f

	lagrange_3_p_2_p_2 = filters_lagrange(L_3_p_here+L_2_p_here+L_2_here)
	l_3_p_2_p_2_f = np.fft.rfft(lagrange_3_p_2_p_2,two_power,norm='ortho')
	s_12_3p_2p_2_f = s12_f_convolve*l_3_p_2_p_2_f

	lagrange_3p= filters_lagrange(L_3_p_here)
	l_3_p_f = np.fft.rfft(lagrange_3p,two_power,norm='ortho')
	s_12_3p_f = s12_f_convolve*l_3_p_f

	lagrange_3_3p= filters_lagrange(L_3_here+L_3_p_here)
	l_3_3_p_f = np.fft.rfft(lagrange_3_3p,two_power,norm='ortho')
	s_31_3_3p_f = s31_f_convolve*l_3_3_p_f

	lagrange_2_3_3_p = filters_lagrange(L_2_here+L_3_here+L_3_p_here)
	l_2_3_3_p_f = np.fft.rfft(lagrange_2_3_3_p,two_power,norm='ortho')
	s_13_2_3_3p_f = s13_f_convolve*l_2_3_3_p_f

	lagrange_2_p_2_3_3_p = filters_lagrange(L_2_p_here+L_2_here+L_3_here+L_3_p_here)
	l_2_p_2_3_3_p_f = np.fft.rfft(lagrange_2_p_2_3_3_p,two_power,norm='ortho')
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

	plt.loglog(f_band,np.abs(x_combo_f_domain))
	plt.show()
	x_combo_val = np.fft.irfft(x_combo_f_domain, norm='ortho')

	plt.semilogy(np.abs(x_combo_val))
	plt.title('time domain')
	plt.show()

	#x_combo_f_domain = np.fft.rfft(window*x_combo_val,norm='ortho')[indices_f_band]
	

	return x_combo_f_domain
'''

'''

def x_combo(L_3_here, L_2_here,L_3_p_here,L_2_p_here):


	lagrange_2= filters_lagrange(L_2_here)
	l_2_f = np.fft.rfft(lagrange_2,two_power,norm='ortho')
	s_13_2_f = s13_f_subtract*l_2_f


	lagrange_2_p_2 = filters_lagrange(L_2_p_here+L_2_here)
	l_2_p_2_f = np.fft.rfft(lagrange_2_p_2,two_power,norm='ortho')
	s_21_2p_2_f = s21_f_subtract*l_2_p_2_f

	lagrange_3_p_2_p_2 = filters_lagrange(L_3_p_here+L_2_p_here+L_2_here)
	l_3_p_2_p_2_f = np.fft.rfft(lagrange_3_p_2_p_2,two_power,norm='ortho')
	s_12_3p_2p_2_f = s12_f_subtract*l_3_p_2_p_2_f

	lagrange_3p= filters_lagrange(L_3_p_here)
	l_3_p_f = np.fft.rfft(lagrange_3p,two_power,norm='ortho')
	s_12_3p_f = s12_f_subtract*l_3_p_f

	lagrange_3_3p= filters_lagrange(L_3_here+L_3_p_here)
	l_3_3_p_f = np.fft.rfft(lagrange_3_3p,two_power,norm='ortho')
	s_31_3_3p_f = s31_f_subtract*l_3_3_p_f

	lagrange_2_3_3_p = filters_lagrange(L_2_here+L_3_here+L_3_p_here)
	l_2_3_3_p_f = np.fft.rfft(lagrange_2_3_3_p,two_power,norm='ortho')
	s_13_2_3_3p_f = s13_f_subtract*l_2_3_3_p_f

	lagrange_2_p_2_3_3_p = filters_lagrange(L_2_p_here+L_2_here+L_3_here+L_3_p_here)
	l_2_p_2_3_3_p_f = np.fft.rfft(lagrange_2_p_2_3_3_p,two_power,norm='ortho')
	tau_21_2_p_2_3_3p_f = tau21_f_subtract*l_2_p_2_3_3_p_f 
	tau_31_2_p_2_3_3p_f = tau31_f_subtract*l_2_p_2_3_3_p_f 

	tau21_3_3p_f = tau21_f_subtract*l_3_3_p_f
	tau31_3_3p_f = tau31_f_subtract*l_3_3_p_f

	tau21_2p_2_f = tau21_f_subtract*l_2_p_2_f
	tau31_2p_2_f = tau31_f_subtract*l_2_p_2_f

	eps_31_2_p_2_3_3p_f = eps31_f_subtract*l_2_p_2_3_3_p_f 

	eps31_3_3p_f = eps31_f_subtract*l_3_3_p_f

	eps31_2p_2_f = eps31_f_subtract*l_2_p_2_f

	eps_21_2_p_2_3_3p_f = eps21_f_subtract*l_2_p_2_3_3_p_f 

	eps21_3_3p_f = eps21_f_subtract*l_3_3_p_f

	eps21_2p_2_f = eps21_f_subtract*l_2_p_2_f

	eps12_3p_f = eps12_f_subtract*l_3_p_f
	tau12_3p_f = tau12_f_subtract*l_3_p_f

	eps12_3p_2p_2_f = eps12_f_subtract*l_3_p_2_p_2_f
	tau12_3p_2p_2_f = tau12_f_subtract*l_3_p_2_p_2_f

	eps13_2_f = eps13_f_subtract*l_2_f
	tau13_2_f = tau13_f_subtract*l_2_f

	eps13_2_3_3p_f = eps13_f_subtract*l_2_3_3_p_f
	tau13_2_3_3p_f = tau13_f_subtract*l_2_3_3_p_f

	x_combo_f_domain = s31_f_subtract + s_13_2_f + s_21_2p_2_f + s_12_3p_2p_2_f - s21_f_subtract - s_12_3p_f - s_31_3_3p_f - s_13_2_3_3p_f + 0.5*(tau_21_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f - tau21_3_3p_f + tau31_3_3p_f - tau21_2p_2_f + tau31_2p_2_f + tau21_f_subtract - tau31_f_subtract) + 0.5*(eps_31_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f + eps31_3_3p_f - tau31_3_3p_f - eps31_2p_2_f + tau31_2p_2_f - eps31_f_subtract + tau31_f_subtract) - 0.5*(eps_21_2_p_2_3_3p_f - tau_21_2_p_2_3_3p_f - eps21_3_3p_f + tau21_3_3p_f + eps21_2p_2_f - tau21_2p_2_f - eps21_f_subtract + tau21_f_subtract) + eps12_3p_f - tau12_3p_f - eps12_3p_2p_2_f + tau12_3p_2p_2_f - eps13_2_f + tau13_2_f + eps13_2_3_3p_f - tau13_2_3_3p_f
	#x_combo_f_domain = s31_f_subtract + s_13_2_f + s_21_2p_2_f + s_12_3p_2p_2_f - s21_f_subtract - s_12_3p_f - s_31_3_3p_f - s_13_2_3_3p_f + 0.5*(tau_21_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f - tau21_3_3p_f + tau31_3_3p_f - tau21_2p_2_f + tau31_2p_2_f + tau21_f_subtract - tau31_f_subtract) + 0.5*(eps_31_2_p_2_3_3p_f - tau_31_2_p_2_3_3p_f + eps31_3_3p_f - tau31_3_3p_f - eps31_2p_2_f + tau31_2p_2_f - eps31_f_subtract + tau31_f_subtract) - 0.5*(eps_21_2_p_2_3_3p_f - tau_21_2_p_2_3_3p_f - eps21_3_3p_f + tau21_3_3p_f + eps21_2p_2_f - tau21_2p_2_f - eps21_f_subtract + tau21_f_subtract) + eps12_3p_f - tau12_3p_f - eps12_3p_2p_2_f + tau12_3p_2p_2_f - eps13_2_f + tau13_2_f + eps13_2_3_3p_f - tau13_2_3_3p_f

	plt.loglog(f_band,np.abs(x_combo_f_domain))
	plt.show()
	x_combo_val = np.fft.irfft(x_combo_f_domain, norm='ortho')

	plt.semilogy(np.abs(x_combo_val))
	plt.title('time domain')
	plt.show()

	#x_combo_f_domain = np.fft.rfft(window*x_combo_val,norm='ortho')[indices_f_band]
	

	return x_combo_f_domain
'''

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

	plt.loglog(f_band,np.abs(x_combo_f_domain))
	plt.show()
	x_combo_val = np.fft.irfft(x_combo_f_domain, norm='ortho')

	plt.semilogy(np.abs(x_combo_val))
	plt.title('time domain')
	plt.show()

	#x_combo_f_domain = np.fft.rfft(window*x_combo_val,norm='ortho')[indices_f_band]
	

	return x_combo_f_domain

def y_combo(L_3_here, L_1_here,L_3_p_here,L_1_p_here):


	lagrange_s_21_3= filters_lagrange(L_3_here)
	s21_3 = fftconvolve(s21,lagrange_s_21_3,'same')


	lagrange_s_23_1_p= filters_lagrange(L_1_p_here)
	s23_1_p = fftconvolve(s23,lagrange_s_23_1_p,'same')



	a = s12+s21_3
	b = s32+s23_1_p
	h = tau32-tau12

	lagrange_b_3_3_p = filters_lagrange((L_3_p_here+L_3_here))
	c = fftconvolve(b,lagrange_b_3_3_p,'same')

	lagrange_a_1_1p = filters_lagrange((L_1_here+L_1_p_here))
	d = fftconvolve(a,lagrange_a_1_1p,'same')

	lagrange_h_3p_3_1_1p = filters_lagrange((L_3_p_here+L_3_here+L_1_here+L_1_p_here))
	alpha = fftconvolve(h,lagrange_h_3p_3_1_1p,'same')

	gamma = fftconvolve(h,lagrange_b_3_3_p,'same')

	beta = fftconvolve(h,lagrange_a_1_1p,'same')

	J = eps12-tau12
	J_three = fftconvolve(J,lagrange_b_3_3_p,'same')
	J_one = fftconvolve(J,lagrange_a_1_1p,'same')
	J_all = fftconvolve(J,lagrange_h_3p_3_1_1p,'same')

	P = eps32-tau32
	P_three = fftconvolve(P,lagrange_b_3_3_p,'same')
	P_one = fftconvolve(P,lagrange_a_1_1p,'same')
	P_all = fftconvolve(P,lagrange_h_3p_3_1_1p,'same')

	Q = eps23-tau23
	Q_one = fftconvolve(Q,lagrange_s_23_1_p,'same')
	Q_three = fftconvolve(Q_one,lagrange_b_3_3_p,'same')

	R = eps21-tau21
	R_one = fftconvolve(R,lagrange_s_21_3,'same')
	R_three = fftconvolve(R_one,lagrange_a_1_1p,'same')


	#TDI X combo RR X_1.5 
	y_combo_val = (a+c)-(b+d)+0.5*(alpha-beta-gamma+h) +0.5*(J_all+J_one-J_three-J) -0.5*(P_all-P_one+P_three-P)+Q_one-Q_three-R_one+R_three

	y_combo_val = y_combo_val[beg_ind:end_ind:]

	y_combo_f_domain = np.fft.rfft(window*y_combo_val,norm='ortho')[indices_f_band]



	return y_combo_f_domain



def z_combo(L_2_here, L_1_here,L_2_p_here,L_1_p_here):


	lagrange_s_32_1= filters_lagrange(L_1_here)
	s32_1 = fftconvolve(s32,lagrange_s_32_1,'same')


	lagrange_s_31_2_p= filters_lagrange(L_2_p_here)
	s31_2_p = fftconvolve(s31,lagrange_s_31_2_p,'same')


	a = s23+s32_1
	b = s13+s31_2_p
	h = tau13-tau23

	lagrange_c = filters_lagrange((L_1_here+L_1_p_here))
	c = fftconvolve(b,lagrange_c,'same')

	lagrange_d = filters_lagrange((L_2_here+L_2_p_here))
	d = fftconvolve(a,lagrange_d,'same')

	lagrange_alpha = filters_lagrange((L_1_p_here+L_1_here+L_2_here+L_2_p_here))
	alpha = fftconvolve(h,lagrange_alpha,'same')

	beta = fftconvolve(h,lagrange_d,'same')

	gamma = fftconvolve(h,lagrange_c,'same')

	J = eps23-tau23
	J_two = fftconvolve(J,lagrange_d,'same')
	J_one = fftconvolve(J,lagrange_c,'same')
	J_all = fftconvolve(J,lagrange_alpha,'same')

	P = eps13-tau13
	P_two = fftconvolve(P,lagrange_d,'same')
	P_one = fftconvolve(P,lagrange_c,'same')
	P_all = fftconvolve(P,lagrange_alpha,'same')

	Q = eps31-tau31
	Q_one = fftconvolve(Q,lagrange_s_31_2_p,'same')
	Q_three = fftconvolve(Q_one,lagrange_c,'same')

	R = eps32-tau32
	R_one = fftconvolve(R,lagrange_s_32_1,'same')
	R_three = fftconvolve(R_one,lagrange_d,'same')

	#TDI X combo RR X_1.5 
	z_combo_val = (a+c)-(b+d)+0.5*(alpha-beta-gamma+h) +0.5*(J_all+J_two-J_one-J) -0.5*(P_all-P_two+P_one-P)+Q_one-Q_three-R_one+R_three

	z_combo_val = z_combo_val[beg_ind:end_ind:]
	#secondary noise power for likelihood calculation

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


	return value,np.sum(log_term_determinant),np.sum(chi_2)	

	
#........................................................................................
#.............................. Raw  PM DATA   ..........................................
#........................................................................................
#Fixed time delays (c = 1)


#simulated delay times in seconds rounded to 9 decimal places
L_1 = 8.339095192
L_1_p = 8.338994879
L_2 = 8.338867041
L_2_p = 8.339095192
L_3 = 8.338994879
L_3_p = 8.338867041


f_s = 4

f_samp = 4


number_n = 29

f_min = 1.0e-4 # (= 0.0009765625)
f_max = 1.0e-1



#............................................................
#................................... DATA . ....................................
#............................................................

data =  np.genfromtxt('./../Data_Simulation/data_fs_4_N=49_FOR_RUN.dat',names=True)
#data =  np.genfromtxt('./../Data_Simulation/data_fs_4_N=49.dat',names=True)



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

'''
s31 = data['s31_LFN']
s21 = data['s21_LFN']
s32 = data['s32_LFN']
s12 = data['s12_LFN']
s23 = data['s23_LFN']
s13 = data['s13_LFN']

tau31 = data['tau31_LFN']
tau21 = data['tau21_LFN']
tau12 = data['tau12_LFN']
tau32 = data['tau32_LFN']
tau23 = data['tau23_LFN']
tau13 = data['tau13_LFN']

eps31 = data['eps31_LFN']
eps21 = data['eps21_LFN']
eps12 = data['eps12_LFN']
eps32 = data['eps32_LFN']
eps23 = data['eps23_LFN']
eps13 = data['eps13_LFN']
'''


length = len(s31)

del data

two_power = next_two_power(length)
m = two_power-length+1

'''
m =length
#avoid circular convolution
two_power=length+m-1
'''
print('length')
print(length)
print('two_power')
print(two_power)

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
window = cosine(length)

'''
#FFT's for manual FFT convolution in interpolation step
s31_f_convolve = np.fft.rfft(window*s31,two_power,norm='ortho')
s21_f_convolve = np.fft.rfft(window*s21,two_power,norm='ortho')
s32_f_convolve = np.fft.rfft(window*s32,two_power,norm='ortho')
s12_f_convolve = np.fft.rfft(window*s12,two_power,norm='ortho')
s23_f_convolve = np.fft.rfft(window*s23,two_power,norm='ortho')
s13_f_convolve = np.fft.rfft(window*s13,two_power,norm='ortho')
tau31_f_convolve = np.fft.rfft(window*tau31,two_power,norm='ortho')
tau21_f_convolve = np.fft.rfft(window*tau21,two_power,norm='ortho')
tau32_f_convolve = np.fft.rfft(window*tau32,two_power,norm='ortho')
tau12_f_convolve = np.fft.rfft(window*tau12,two_power,norm='ortho')
tau23_f_convolve = np.fft.rfft(window*tau23,two_power,norm='ortho')
tau13_f_convolve = np.fft.rfft(window*tau13,two_power,norm='ortho')
eps31_f_convolve = np.fft.rfft(window*eps31,two_power,norm='ortho')
eps21_f_convolve = np.fft.rfft(window*eps21,two_power,norm='ortho')
eps32_f_convolve = np.fft.rfft(window*eps32,two_power,norm='ortho')
eps12_f_convolve = np.fft.rfft(window*eps12,two_power,norm='ortho')
eps23_f_convolve = np.fft.rfft(window*eps23,two_power,norm='ortho')
eps13_f_convolve = np.fft.rfft(window*eps13,two_power,norm='ortho')
'''

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
'''
#FFTs for TDI subtraction (Zeroes have to be padded on either side instead of at end.)
s31_half_pad = np.pad(window*s31,(half_extra,half_extra),'constant')
s21_half_pad = np.pad(window*s21,(half_extra,half_extra),'constant')
s32_half_pad = np.pad(window*s32,(half_extra,half_extra),'constant')
s12_half_pad = np.pad(window*s12,(half_extra,half_extra),'constant')
s23_half_pad = np.pad(window*s23,(half_extra,half_extra),'constant')
s13_half_pad = np.pad(window*s13,(half_extra,half_extra),'constant')
tau31_half_pad = np.pad(window*tau31,(half_extra,half_extra),'constant')
tau21_half_pad = np.pad(window*tau21,(half_extra,half_extra),'constant')
tau32_half_pad = np.pad(window*tau32,(half_extra,half_extra),'constant')
tau12_half_pad = np.pad(window*tau12,(half_extra,half_extra),'constant')
tau23_half_pad = np.pad(window*tau23,(half_extra,half_extra),'constant')
tau13_half_pad = np.pad(window*tau13,(half_extra,half_extra),'constant')
eps31_half_pad = np.pad(window*eps31,(half_extra,half_extra),'constant')
eps21_half_pad = np.pad(window*eps21,(half_extra,half_extra),'constant')
eps32_half_pad = np.pad(window*eps32,(half_extra,half_extra),'constant')
eps12_half_pad = np.pad(window*eps12,(half_extra,half_extra),'constant')
eps23_half_pad = np.pad(window*eps23,(half_extra,half_extra),'constant')
eps13_half_pad = np.pad(window*eps13,(half_extra,half_extra),'constant')
'''

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
'''
window = cosine(two_power)
s31_f_subtract = np.fft.rfft(window*s31_half_pad,norm='ortho')
s21_f_subtract = np.fft.rfft(window*s21_half_pad,norm='ortho')
s32_f_subtract = np.fft.rfft(window*s32_half_pad,norm='ortho')
s12_f_subtract = np.fft.rfft(window*s12_half_pad,norm='ortho')
s23_f_subtract = np.fft.rfft(window*s23_half_pad,norm='ortho')
s13_f_subtract = np.fft.rfft(window*s13_half_pad,norm='ortho')
tau31_f_subtract = np.fft.rfft(window*tau31_half_pad,norm='ortho')
tau21_f_subtract = np.fft.rfft(window*tau21_half_pad,norm='ortho')
tau32_f_subtract = np.fft.rfft(window*tau32_half_pad,norm='ortho')
tau12_f_subtract = np.fft.rfft(window*tau12_half_pad,norm='ortho')
tau23_f_subtract = np.fft.rfft(window*tau23_half_pad,norm='ortho')
tau13_f_subtract = np.fft.rfft(window*tau13_half_pad,norm='ortho')
eps31_f_subtract = np.fft.rfft(window*eps31_half_pad,norm='ortho')
eps21_f_subtract = np.fft.rfft(window*eps21_half_pad,norm='ortho')
eps32_f_subtract = np.fft.rfft(window*eps32_half_pad,norm='ortho')
eps12_f_subtract = np.fft.rfft(window*eps12_half_pad,norm='ortho')
eps23_f_subtract = np.fft.rfft(window*eps23_half_pad,norm='ortho')
eps13_f_subtract = np.fft.rfft(window*eps13_half_pad,norm='ortho')
'''

'''
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
'''

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

'''
beg_ind,end_ind = cut_data(L_3,L_2,L_1,L_3_p,L_2_p,L_1_p,f_s,length)
window = cosine(length)[beg_ind:end_ind:]
cut_data_length = len(window)
#secondary noise power for likelihood calculation
f_band = np.fft.rfftfreq(len(window),1/f_s)
#p_n = secondary_noise_power(f_band)
indices_f_band = np.where(np.logical_and(f_band>=f_min, f_band<=f_max))
f_band=f_band[indices_f_band]
'''
f_band = np.fft.rfftfreq(two_power,1/f_s)

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
