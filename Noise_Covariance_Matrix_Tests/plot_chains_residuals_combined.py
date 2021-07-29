import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpmath import binomial,log,pi,ceil
import corner 
import matplotlib.lines as mlines
from chainconsumer import ChainConsumer
from astropy import constants as const
from scipy.signal import fftconvolve,cosine
import math
import matplotlib as mpl
import matplotlib.colors as color 
from matplotlib import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (7,7)
#For adding table to subplot in corner plot
#mpl.rcParams["text.usetex"] = True
#mpl.rcParams["text.latex.preamble"].append(r'\usepackage{tabularx}')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


#cutting off bad data due to delay times and filter length
def cut_data(D_3,D_2,D_1,D_3_p,D_2_p,D_1_p,f_rate,length):

	D_2_val = int(round(f_rate*D_2))
	D_3_val = int(round(f_rate*D_3))
	D_1_val = int(round(f_rate*D_1))

	D_2_p_val = int(round(f_rate*D_2_p))
	D_3_p_val = int(round(f_rate*D_3_p))
	D_1_p_val = int(round(f_rate*D_1_p))

	filter_cut = int(round((number_n-1)))

	beg_ind = filter_cut+D_3_val+D_2_val++D_1_val+D_3_p_val+D_2_p_val+D_1_p_val
	#beg_ind = filter_cut
	end_ind = int(length-filter_cut-1)


	return beg_ind, end_ind


#...........................CREATING FILTERS.......................................
def filters_lagrange(delay):

	D = delay*f_samp
	i, d_frac = divmod(D,1)

	if d_frac >= 0.5:
		i+=1
		d_frac = -1*(1-d_frac)

	# delayed filters we're convolving with

	lagrange_filter = np.zeros(len(h_points))
	h_points_filter = h_points - int(i) + 1
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


def x_combo(L_3_here, L_2_here,L_3_p_here,L_2_p_here):


	lagrange_s_13_2= filters_lagrange(L_2_here)
	s_13_2 = fftconvolve(s13,lagrange_s_13_2,'same')


	lagrange_s_12_3p= filters_lagrange(L_3_p_here)
	s_12_3p = fftconvolve(s12,lagrange_s_12_3p,'same')

	#DEFINITELY SEE page 367 in notes


	G = s21+s_12_3p
	F = s31+s_13_2
	H = tau21-tau31

	lagrange_A_2p_2 = filters_lagrange((L_2_p_here+L_2_here))
	A = fftconvolve(G,lagrange_A_2p_2,'same')

	lagrange_B_3_3p = filters_lagrange((L_3_here+L_3_p_here))
	B = fftconvolve(F,lagrange_B_3_3p,'same')

	lagrange_C_2p_2_3_3p = filters_lagrange((L_2_p_here+L_2_here+L_3_here+L_3_p_here))
	C = fftconvolve(H,lagrange_C_2p_2_3_3p,'same')

	#lagrange_D_3_3p = filters_lagrange((L_3_here+L_3_p_here))
	D_equation = fftconvolve(H,lagrange_B_3_3p,'same')

	#lagrange_E_2p_2 = filters_lagrange((L_2_p_here+L_2_here))
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
	'''
	plt.semilogy(x_combo_val,label = 'TDI X')
	#plt.semilogy(s31-s21,label = 's31-s21',alpha = 0.5)
	plt.legend()
	plt.title('x combo')
	plt.show()
	'''
	x_combo_val = x_combo_val[beg_ind:end_ind:]

	x_combo_f_domain = np.fft.rfft(window*x_combo_val,norm='ortho')[indices_f_band]
	'''
	f_here = np.fft.rfftfreq(int(len(x_combo_f_domain)*2-1),1/f_s)
	plt.loglog(f_here,np.abs(x_combo_f_domain),label = 'TDI X frequency domain')
	#plt.semilogy(s31-s21,label = 's31-s21',alpha = 0.5)
	plt.legend()
	plt.title('x combo f domain')
	plt.show()
	'''

	return x_combo_f_domain



def y_combo(L_3_here, L_1_here,L_3_p_here,L_1_p_here):


	lagrange_s_21_3= filters_lagrange(L_3_here)
	s21_3 = fftconvolve(s21,lagrange_s_21_3,'same')


	lagrange_s_23_1_p= filters_lagrange(L_1_p_here)
	s23_1_p = fftconvolve(s23,lagrange_s_23_1_p,'same')

	#DEFINITELY SEE page 367 and 394 (y channel) in notes


	a = s12+s21_3
	b = s32+s23_1_p
	h = tau32-tau12

	lagrange_b_3_3_p = filters_lagrange((L_3_p_here+L_3_here))
	c = fftconvolve(b,lagrange_b_3_3_p,'same')

	lagrange_a_1_1p = filters_lagrange((L_1_here+L_1_p_here))
	d = fftconvolve(a,lagrange_a_1_1p,'same')

	lagrange_h_3p_3_1_1p = filters_lagrange((L_3_p_here+L_3_here+L_1_here+L_1_p_here))
	alpha = fftconvolve(h,lagrange_h_3p_3_1_1p,'same')

	#lagrange_h_3p_3 = filters_lagrange((L_3_p_here+L_3_here))
	gamma = fftconvolve(h,lagrange_b_3_3_p,'same')

	#lagrange_h_1_1p = filters_lagrange((L_1_here+L_1_p_here))
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

	#DEFINITELY SEE page 367 and 394 (y channel) in notes


	a = s23+s32_1
	b = s13+s31_2_p
	h = tau13-tau23

	lagrange_c = filters_lagrange((L_1_here+L_1_p_here))
	c = fftconvolve(b,lagrange_c,'same')

	lagrange_d = filters_lagrange((L_2_here+L_2_p_here))
	d = fftconvolve(a,lagrange_d,'same')

	lagrange_alpha = filters_lagrange((L_1_p_here+L_1_here+L_2_here+L_2_p_here))
	alpha = fftconvolve(h,lagrange_alpha,'same')

	#lagrange_beta = filters_lagrange((L_2_here+L_2_p_here))
	beta = fftconvolve(h,lagrange_d,'same')

	#lagrange_gamma = filters_lagrange((L_1_p_here+L_1_here))
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



def x_combo_LFN(L_3_here, L_2_here,L_3_p_here,L_2_p_here):


	lagrange_s_13_2= filters_lagrange(L_2_here)
	s_13_2 = fftconvolve(s13_LFN,lagrange_s_13_2,'same')


	lagrange_s_12_3p= filters_lagrange(L_3_p_here)
	s_12_3p = fftconvolve(s12_LFN,lagrange_s_12_3p,'same')

	#DEFINITELY SEE page 367 in notes


	G = s21_LFN+s_12_3p
	F = s31_LFN+s_13_2
	H = tau21_LFN-tau31_LFN

	lagrange_A_2p_2 = filters_lagrange((L_2_p_here+L_2_here))
	A = fftconvolve(G,lagrange_A_2p_2,'same')

	lagrange_B_3_3p = filters_lagrange((L_3_here+L_3_p_here))
	B = fftconvolve(F,lagrange_B_3_3p,'same')

	lagrange_C_2p_2_3_3p = filters_lagrange((L_2_p_here+L_2_here+L_3_here+L_3_p_here))
	C = fftconvolve(H,lagrange_C_2p_2_3_3p,'same')

	lagrange_D_3_3p = filters_lagrange((L_3_here+L_3_p_here))
	D_equation = fftconvolve(H,lagrange_D_3_3p,'same')

	lagrange_E_2p_2 = filters_lagrange((L_2_p_here+L_2_here))
	E = fftconvolve(H,lagrange_E_2p_2,'same')

	J = eps31_LFN-tau31_LFN
	J_three = fftconvolve(J,lagrange_B_3_3p,'same')
	J_two = fftconvolve(J,lagrange_A_2p_2,'same')
	J_all = fftconvolve(J,lagrange_C_2p_2_3_3p,'same')

	P = eps21_LFN-tau21_LFN
	P_three = fftconvolve(P,lagrange_B_3_3p,'same')
	P_two = fftconvolve(P,lagrange_A_2p_2,'same')
	P_all = fftconvolve(P,lagrange_C_2p_2_3_3p,'same')

	Q = eps12_LFN-tau12_LFN
	Q_one = fftconvolve(Q,lagrange_s_12_3p,'same')
	Q_three = fftconvolve(Q_one,lagrange_A_2p_2,'same')

	R = eps13_LFN-tau13_LFN
	R_one = fftconvolve(R,lagrange_s_13_2,'same')
	R_three = fftconvolve(R_one,lagrange_B_3_3p,'same')



	#TDI X combo RR X_1.5 split int
	x_combo_val = (F+A)-(G+B)+0.5*(C-D_equation-E+H) +0.5*(J_all+J_three-J_two-J) -0.5*(P_all-P_three+P_two-P)+Q_one-Q_three-R_one+R_three

	#TDI X combo RR X_1.5 
	#x_combo_val = (F+A)-(G+B)+0.5*(C-D_equation-E+H)

	x_combo_val = x_combo_val[beg_ind:end_ind:]

	
	x_combo_f_domain = np.fft.rfft(window*x_combo_val,norm='ortho')[indices_f_band]
	

	return x_combo_f_domain



def y_combo_LFN(L_3_here, L_1_here,L_3_p_here,L_1_p_here):


	lagrange_s_21_3= filters_lagrange(L_3_here)
	s21_3 = fftconvolve(s21_LFN,lagrange_s_21_3,'same')


	lagrange_s_23_1_p= filters_lagrange(L_1_p_here)
	s23_1_p = fftconvolve(s23_LFN,lagrange_s_23_1_p,'same')

	#DEFINITELY SEE page 367 and 394 (y channel) in notes


	a = s12_LFN+s21_3
	b = s32_LFN+s23_1_p
	h = tau32_LFN-tau12_LFN

	lagrange_b_3_3_p = filters_lagrange((L_3_p_here+L_3_here))
	c = fftconvolve(b,lagrange_b_3_3_p,'same')

	lagrange_a_1_1p = filters_lagrange((L_1_here+L_1_p_here))
	d = fftconvolve(a,lagrange_a_1_1p,'same')

	lagrange_h_3p_3_1_1p = filters_lagrange((L_3_p_here+L_3_here+L_1_here+L_1_p_here))
	alpha = fftconvolve(h,lagrange_h_3p_3_1_1p,'same')

	lagrange_h_3p_3 = filters_lagrange((L_3_p_here+L_3_here))
	gamma = fftconvolve(h,lagrange_h_3p_3,'same')

	lagrange_h_1_1p = filters_lagrange((L_1_here+L_1_p_here))
	beta = fftconvolve(h,lagrange_h_1_1p,'same')

	J = eps12_LFN-tau12_LFN
	J_three = fftconvolve(J,lagrange_b_3_3_p,'same')
	J_one = fftconvolve(J,lagrange_a_1_1p,'same')
	J_all = fftconvolve(J,lagrange_h_3p_3_1_1p,'same')

	P = eps32_LFN-tau32_LFN
	P_three = fftconvolve(P,lagrange_b_3_3_p,'same')
	P_one = fftconvolve(P,lagrange_a_1_1p,'same')
	P_all = fftconvolve(P,lagrange_h_3p_3_1_1p,'same')

	Q = eps23_LFN-tau23_LFN
	Q_one = fftconvolve(Q,lagrange_s_23_1_p,'same')
	Q_three = fftconvolve(Q_one,lagrange_b_3_3_p,'same')

	R = eps21_LFN-tau21_LFN
	R_one = fftconvolve(R,lagrange_s_21_3,'same')
	R_three = fftconvolve(R_one,lagrange_a_1_1p,'same')


	#TDI X combo RR X_1.5 
	y_combo_val = (a+c)-(b+d)+0.5*(alpha-beta-gamma+h) +0.5*(J_all+J_one-J_three-J) -0.5*(P_all-P_one+P_three-P)+Q_one-Q_three-R_one+R_three
	

	y_combo_val = y_combo_val[beg_ind:end_ind:]
	

	
	y_combo_f_domain = np.fft.rfft(window*y_combo_val,norm='ortho')[indices_f_band]
	

	return y_combo_f_domain



def z_combo_LFN(L_2_here, L_1_here,L_2_p_here,L_1_p_here):


	lagrange_s_32_1= filters_lagrange(L_1_here)
	s32_1 = fftconvolve(s32_LFN,lagrange_s_32_1,'same')


	lagrange_s_31_2_p= filters_lagrange(L_2_p_here)
	s31_2_p = fftconvolve(s31_LFN,lagrange_s_31_2_p,'same')

	#DEFINITELY SEE page 367 and 394 (y channel) in notes


	a = s23_LFN+s32_1
	b = s13_LFN+s31_2_p
	h = tau13_LFN-tau23_LFN

	lagrange_c = filters_lagrange((L_1_here+L_1_p_here))
	c = fftconvolve(b,lagrange_c,'same')

	lagrange_d = filters_lagrange((L_2_here+L_2_p_here))
	d = fftconvolve(a,lagrange_d,'same')

	lagrange_alpha = filters_lagrange((L_1_p_here+L_1_here+L_2_here+L_2_p_here))
	alpha = fftconvolve(h,lagrange_alpha,'same')

	lagrange_beta = filters_lagrange((L_2_here+L_2_p_here))
	beta = fftconvolve(h,lagrange_beta,'same')

	lagrange_gamma = filters_lagrange((L_1_p_here+L_1_here))
	gamma = fftconvolve(h,lagrange_gamma,'same')

	J = eps23_LFN-tau23_LFN
	J_two = fftconvolve(J,lagrange_d,'same')
	J_one = fftconvolve(J,lagrange_c,'same')
	J_all = fftconvolve(J,lagrange_alpha,'same')

	P = eps13_LFN-tau13_LFN
	P_two = fftconvolve(P,lagrange_d,'same')
	P_one = fftconvolve(P,lagrange_c,'same')
	P_all = fftconvolve(P,lagrange_alpha,'same')

	Q = eps31_LFN-tau31_LFN
	Q_one = fftconvolve(Q,lagrange_s_31_2_p,'same')
	Q_three = fftconvolve(Q_one,lagrange_c,'same')

	R = eps32_LFN-tau32_LFN
	R_one = fftconvolve(R,lagrange_s_32_1,'same')
	R_three = fftconvolve(R_one,lagrange_d,'same')

	#TDI X combo RR X_1.5 
	z_combo_val = (a+c)-(b+d)+0.5*(alpha-beta-gamma+h) +0.5*(J_all+J_two-J_one-J) -0.5*(P_all-P_two+P_one-P)+Q_one-Q_three-R_one+R_three
	

	

	z_combo_val = z_combo_val[beg_ind:end_ind:]
	#secondary noise power for likelihood calculation

	
	z_combo_f_domain = np.fft.rfft(window*z_combo_val,norm='ortho')[indices_f_band]
	

	

	return z_combo_f_domain



def x_combo_noise(L_3_here, L_2_here,L_3_p_here,L_2_p_here):


	lagrange_s_13_2= filters_lagrange(L_2_here)
	s_13_2 = fftconvolve(s13_noise,lagrange_s_13_2,'same')


	lagrange_s_12_3p= filters_lagrange(L_3_p_here)
	s_12_3p = fftconvolve(s12_noise,lagrange_s_12_3p,'same')

	#DEFINITELY SEE page 367 in notes


	G = s21_noise+s_12_3p
	F = s31_noise+s_13_2
	H = tau21_noise-tau31_noise

	lagrange_A_2p_2 = filters_lagrange((L_2_p_here+L_2_here))
	A = fftconvolve(G,lagrange_A_2p_2,'same')

	lagrange_B_3_3p = filters_lagrange((L_3_here+L_3_p_here))
	B = fftconvolve(F,lagrange_B_3_3p,'same')

	lagrange_C_2p_2_3_3p = filters_lagrange((L_2_p_here+L_2_here+L_3_here+L_3_p_here))
	C = fftconvolve(H,lagrange_C_2p_2_3_3p,'same')

	lagrange_D_3_3p = filters_lagrange((L_3_here+L_3_p_here))
	D_equation = fftconvolve(H,lagrange_D_3_3p,'same')

	lagrange_E_2p_2 = filters_lagrange((L_2_p_here+L_2_here))
	E = fftconvolve(H,lagrange_E_2p_2,'same')

	J = eps31_noise-tau31_noise
	J_three = fftconvolve(J,lagrange_B_3_3p,'same')
	J_two = fftconvolve(J,lagrange_A_2p_2,'same')
	J_all = fftconvolve(J,lagrange_C_2p_2_3_3p,'same')

	P = eps21_noise-tau21_noise
	P_three = fftconvolve(P,lagrange_B_3_3p,'same')
	P_two = fftconvolve(P,lagrange_A_2p_2,'same')
	P_all = fftconvolve(P,lagrange_C_2p_2_3_3p,'same')

	Q = eps12_noise-tau12_noise
	Q_one = fftconvolve(Q,lagrange_s_12_3p,'same')
	Q_three = fftconvolve(Q_one,lagrange_A_2p_2,'same')

	R = eps13_noise-tau13_noise
	R_one = fftconvolve(R,lagrange_s_13_2,'same')
	R_three = fftconvolve(R_one,lagrange_B_3_3p,'same')



	#TDI X combo RR X_1.5 
	x_combo_val = (F+A)-(G+B)+0.5*(C-D_equation-E+H) +0.5*(J_all+J_three-J_two-J) -0.5*(P_all-P_three+P_two-P)+Q_one-Q_three-R_one+R_three


	x_combo_val = x_combo_val[beg_ind:end_ind:]

	x_combo_f_domain = np.fft.rfft(window*x_combo_val,norm='ortho')[indices_f_band]
	

	return x_combo_f_domain



def y_combo_noise(L_3_here, L_1_here,L_3_p_here,L_1_p_here):


	lagrange_s_21_3= filters_lagrange(L_3_here)
	s21_3 = fftconvolve(s21_noise,lagrange_s_21_3,'same')


	lagrange_s_23_1_p= filters_lagrange(L_1_p_here)
	s23_1_p = fftconvolve(s23_noise,lagrange_s_23_1_p,'same')

	#DEFINITELY SEE page 367 and 394 (y channel) in notes


	a = s12_noise+s21_3
	b = s32_noise+s23_1_p
	h = tau32_noise-tau12_noise

	lagrange_b_3_3_p = filters_lagrange((L_3_p_here+L_3_here))
	c = fftconvolve(b,lagrange_b_3_3_p,'same')

	lagrange_a_1_1p = filters_lagrange((L_1_here+L_1_p_here))
	d = fftconvolve(a,lagrange_a_1_1p,'same')

	lagrange_h_3p_3_1_1p = filters_lagrange((L_3_p_here+L_3_here+L_1_here+L_1_p_here))
	alpha = fftconvolve(h,lagrange_h_3p_3_1_1p,'same')

	lagrange_h_3p_3 = filters_lagrange((L_3_p_here+L_3_here))
	gamma = fftconvolve(h,lagrange_h_3p_3,'same')

	lagrange_h_1_1p = filters_lagrange((L_1_here+L_1_p_here))
	beta = fftconvolve(h,lagrange_h_1_1p,'same')

	J = eps12_noise-tau12_noise
	J_three = fftconvolve(J,lagrange_b_3_3_p,'same')
	J_one = fftconvolve(J,lagrange_a_1_1p,'same')
	J_all = fftconvolve(J,lagrange_h_3p_3_1_1p,'same')

	P = eps32_noise-tau32_noise
	P_three = fftconvolve(P,lagrange_b_3_3_p,'same')
	P_one = fftconvolve(P,lagrange_a_1_1p,'same')
	P_all = fftconvolve(P,lagrange_h_3p_3_1_1p,'same')

	Q = eps23_noise-tau23_noise
	Q_one = fftconvolve(Q,lagrange_s_23_1_p,'same')
	Q_three = fftconvolve(Q_one,lagrange_b_3_3_p,'same')

	R = eps21_noise-tau21_noise
	R_one = fftconvolve(R,lagrange_s_21_3,'same')
	R_three = fftconvolve(R_one,lagrange_a_1_1p,'same')


	#TDI X combo RR X_1.5 
	y_combo_val = (a+c)-(b+d)+0.5*(alpha-beta-gamma+h) +0.5*(J_all+J_one-J_three-J) -0.5*(P_all-P_one+P_three-P)+Q_one-Q_three-R_one+R_three
	

	
	y_combo_val = y_combo_val[beg_ind:end_ind:]
	

	
	y_combo_f_domain = np.fft.rfft(window*y_combo_val,norm='ortho')[indices_f_band]
	


	return y_combo_f_domain



def z_combo_noise(L_2_here, L_1_here,L_2_p_here,L_1_p_here):


	lagrange_s_32_1= filters_lagrange(L_1_here)
	s32_1 = fftconvolve(s32_noise,lagrange_s_32_1,'same')


	lagrange_s_31_2_p= filters_lagrange(L_2_p_here)
	s31_2_p = fftconvolve(s31_noise,lagrange_s_31_2_p,'same')

	#DEFINITELY SEE page 367 and 394 (y channel) in notes


	a = s23_noise+s32_1
	b = s13_noise+s31_2_p
	h = tau13_noise-tau23_noise

	lagrange_c = filters_lagrange((L_1_here+L_1_p_here))
	c = fftconvolve(b,lagrange_c,'same')

	lagrange_d = filters_lagrange((L_2_here+L_2_p_here))
	d = fftconvolve(a,lagrange_d,'same')

	lagrange_alpha = filters_lagrange((L_1_p_here+L_1_here+L_2_here+L_2_p_here))
	alpha = fftconvolve(h,lagrange_alpha,'same')

	lagrange_beta = filters_lagrange((L_2_here+L_2_p_here))
	beta = fftconvolve(h,lagrange_beta,'same')

	lagrange_gamma = filters_lagrange((L_1_p_here+L_1_here))
	gamma = fftconvolve(h,lagrange_gamma,'same')

	J = eps23_noise-tau23_noise
	J_two = fftconvolve(J,lagrange_d,'same')
	J_one = fftconvolve(J,lagrange_c,'same')
	J_all = fftconvolve(J,lagrange_alpha,'same')

	P = eps13_noise-tau13_noise
	P_two = fftconvolve(P,lagrange_d,'same')
	P_one = fftconvolve(P,lagrange_c,'same')
	P_all = fftconvolve(P,lagrange_alpha,'same')

	Q = eps31_noise-tau31_noise
	Q_one = fftconvolve(Q,lagrange_s_31_2_p,'same')
	Q_three = fftconvolve(Q_one,lagrange_c,'same')

	R = eps32_noise-tau32_noise
	R_one = fftconvolve(R,lagrange_s_32_1,'same')
	R_three = fftconvolve(R_one,lagrange_d,'same')

	#TDI X combo RR X_1.5 
	z_combo_val = (a+c)-(b+d)+0.5*(alpha-beta-gamma+h) +0.5*(J_all+J_two-J_one-J) -0.5*(P_all-P_two+P_one-P)+Q_one-Q_three-R_one+R_three


	z_combo_val = z_combo_val[beg_ind:end_ind:]
	#secondary noise power for likelihood calculation
	

	
	z_combo_f_domain = np.fft.rfft(window*z_combo_val,norm='ortho')[indices_f_band]
		
	return z_combo_f_domain


def covariance_equal_arm(delay_array):

	#zero_array = np.zeros(len(f_band))
	#avg_L = np.mean(delay_array)
	#avg_L = L_arm/const.c.value
	a = 16*np.power(np.sin(2*np.pi*f_band*avg_L),2)*Sy_OP+(8*np.power(np.sin(4*np.pi*f_band*avg_L),2)+32*np.power(np.sin(2*np.pi*f_band*avg_L),2))*Sy_PM

	b = -4*np.sin(2*np.pi*f_band*avg_L)*np.sin(4*np.pi*f_band*avg_L)*(4*Sy_PM+Sy_OP)

	cov = np.array([[a,b,b],[b,a,b],[b,b,a]])

	return 2*a,2*b


def covariance_unequal_arm(delay_array):
#See page 542-543 
#Derivations Begin Page 492. Still Based off of (Estabrook et al 2000) (as done in Equal-arm case, 
#re-derived acceleration noise components from Rigid Rotation TDI equation in (Tinto et al. 2004)

	'''
	#sanity checking using avg_L for all parametrrs 
	avg_L = np.mean(delay_array)
	L1 = avg_L
	L1p = avg_L
	L2 = avg_L
	L2p = avg_L
	L3 = avg_L
	L3p = avg_L
	'''

	
	L1 = delay_array[0]
	L1p = delay_array[1]
	L2 = delay_array[2]
	L2p = delay_array[3]
	L3 = delay_array[4]
	L3p = delay_array[5]
	

	# diagonal elements optical path component
	xx_OP = 4*Sy_OP*(2-np.cos(2*np.pi*f_band*(L3+L3p))-np.cos(2*np.pi*f_band*(L2+L2p)))
	yy_OP = 4*Sy_OP*(2-np.cos(2*np.pi*f_band*(L1+L1p))-np.cos(2*np.pi*f_band*(L3+L3p)))
	zz_OP = 4*Sy_OP*(2-np.cos(2*np.pi*f_band*(L2+L2p))-np.cos(2*np.pi*f_band*(L1+L1p)))

	# diagonal elements proof mass component
	xx_PM = -4*Sy_PM*(-6+2*np.cos(2*np.pi*f_band*(L2+L2p))+np.cos(2*np.pi*f_band*(L2+L2p-L3-L3p))+2*np.cos(2*np.pi*f_band*(L3+L3p))+np.cos(2*np.pi*f_band*(L2+L2p+L3+L3p)))
	yy_PM = -4*Sy_PM*(-6+2*np.cos(2*np.pi*f_band*(L1+L1p))+np.cos(2*np.pi*f_band*(L1+L1p-L3-L3p))+2*np.cos(2*np.pi*f_band*(L3+L3p))+np.cos(2*np.pi*f_band*(L1+L1p+L3+L3p)))
	zz_PM = -4*Sy_PM*(-6+2*np.cos(2*np.pi*f_band*(L1+L1p))+np.cos(2*np.pi*f_band*(L1+L1p-L2-L2p))+2*np.cos(2*np.pi*f_band*(L2+L2p))+np.cos(2*np.pi*f_band*(L1+L1p+L2+L2p)))


	#add proof-mass and optical path components together
	xx = xx_OP + xx_PM
	yy = yy_OP + yy_PM
	zz = zz_OP + zz_PM

	#off-diagonal elements  
	xy = (4*Sy_PM+Sy_OP)*np.exp(-2j*np.pi*f_band*(L1+L1p+L3))*(-1+np.exp(2j*np.pi*f_band*(L1+L1p)))*(-1+np.exp(2j*np.pi*f_band*(L2+L2p)))*(1+np.exp(2j*np.pi*f_band*(L3+L3p)))
	#yx = (4*Sy_PM+Sy_OP)*np.exp(-2j*np.pi*f_band*(L2+L2p+L3p))*(-1+np.exp(2j*np.pi*f_band*(L1+L1p)))*(-1+np.exp(2j*np.pi*f_band*(L2+L2p)))*(1+np.exp(2j*np.pi*f_band*(L3+L3p)))

	xz = (4*Sy_PM+Sy_OP)*np.exp(-2j*np.pi*f_band*(L1+L1p+L2p))*(-1+np.exp(2j*np.pi*f_band*(L1+L1p)))*(1+np.exp(2j*np.pi*f_band*(L2+L2p)))*(-1+np.exp(2j*np.pi*f_band*(L3+L3p)))
	#zx = (4*Sy_PM+Sy_OP)*np.exp(-2j*np.pi*f_band*(L2+L3+L3p))*(-1+np.exp(2j*np.pi*f_band*(L1+L1p)))*(1+np.exp(2j*np.pi*f_band*(L2+L2p)))*(-1+np.exp(2j*np.pi*f_band*(L3+L3p)))

	yz  = (4*Sy_PM+Sy_OP)*np.exp(-2j*np.pi*f_band*(L1+L2+L2p))*(1+np.exp(2j*np.pi*f_band*(L1+L1p)))*(-1+np.exp(2j*np.pi*f_band*(L2+L2p)))*(-1+np.exp(2j*np.pi*f_band*(L3+L3p)))
	#zy  = (4*Sy_PM+Sy_OP)*np.exp(-2j*np.pi*f_band*(L1p+L3+L3p))*(1+np.exp(2j*np.pi*f_band*(L1+L1p)))*(-1+np.exp(2j*np.pi*f_band*(L2+L2p)))*(-1+np.exp(2j*np.pi*f_band*(L3+L3p)))



	#cov = np.array([[xx,xy,xz],[yx,yy,yz],[zx,zy,zz]])
	#cov = np.array([[xx,yx,zx],[xy,yy,zy],[xz,yz,zz]])
	cov = np.array([[xx,xy,xz],[np.conj(xy),yy,yz],[np.conj(xz),np.conj(yz),zz]])



	m = xx
	n = yy
	p = zz

	
	ax = np.real(xy)
	ay = np.imag(xy)
	bx = np.real(xz)
	by = np.imag(xz)
	cx = np.real(yz)
	cy = np.imag(yz)



	return 2*m,2*n,2*p,2*ax,2*ay,2*bx,2*by,2*cx,2*cy


#See pages 609-613
def likelihood_analytical_equal_arm(x_combo_f,y_combo_f,z_combo_f,delay_array,point_estimate):

	x_x = np.real(x_combo_f)
	x_y = np.imag(x_combo_f)

	y_x = np.real(y_combo_f)
	y_y = np.imag(y_combo_f)

	z_x = np.real(z_combo_f)
	z_y = np.imag(z_combo_f)

	#covariance_here = covariance_S_n_f(f_band).T


	
	a,b = covariance_equal_arm(delay_array)

	#Needed in inverse calculation
	A_ = a**2 - b**2
	B_ = b**2 - a*b

	#det_here = np.array(np.linalg.det(np.pi*covariance_here),dtype=np.complex128)
	determinant = a*A_+2*b*B_
	log_term_determinant = np.log(determinant)
	


	chi_2 = 1/determinant*(A_*(x_x**2+x_y**2+y_x**2+y_y**2+z_x**2+z_y**2) + 2*B_*(x_x*y_x+x_y*y_y+x_x*z_x+x_y*z_y+y_x*z_x+y_y*z_y))



	
	#print(np.sum(log_term_determinant))
	#chi_2 = conj_x*inv_T[0][0][:]*x_combo_f + conj_y*inv_T[1][1][:]*y_combo_f + conj_z*inv_T[2][2][:]*z_combo_f + conj_x*inv_T[0][1][:]*y_combo_f + conj_y*inv_T[1][0][:]*x_combo_f + conj_y*inv_T[1][2][:]*z_combo_f + conj_z*inv_T[2][1][:]*y_combo_f + conj_x*inv_T[0][2][:]*z_combo_f + conj_z*inv_T[2][0][:]*x_combo_f

	print('log term determinant')
	print(log_term_determinant)
	print('np.sum(log_term_determinant)')

	print(np.sum(log_term_determinant))
	print('np.sum(chi_2)')
	print(np.sum(chi_2))
	#value = -1/2*(np.sum(chi_2))-np.sum(log_term_factor)
	#value = -1*np.sum(chi_2) - np.sum(log_term_determinant)
	value = -1*np.sum(chi_2) - log_term_factor - np.sum(log_term_determinant) 

	plt.loglog(f_band,np.abs(x_combo_f)**2,label='Resiudual XX')
	plt.loglog(f_band,np.abs(x_combo_f)*np.abs(y_combo_f),label='Residual XY')
	#plt.loglog(f_band,np.abs(A)**2,label='Residual A Channel',alpha=0.5)
	#plt.loglog(f_band,np.abs(E)**2,label='Residual E Channel',alpha=0.5)
	#plt.loglog(f_band,np.abs(T)**2,label='Residual T Channel',alpha=0.5)
	#plt.loglog(f_band,m,label=r'$\Sigma_{UNEQUAL}[0][0]$')
	#plt.loglog(f_band,np.abs(ax),label=r'$\Sigma_{UNEQUAL}[0][1]$')
	plt.loglog(f_band,a,label=r'$\Sigma_{EQUAL}[0][0]$')
	plt.loglog(f_band,b,label=r'$\Sigma_{EQUAL}[0][1]$')
	#plt.loglog(f_band,noise_A_result,label=r'$\Sigma_{AET}[0][0]$')
	#plt.loglog(f_band,noise_T_result,label=r'$\Sigma_{AET}[2][2]$')
	#plt.loglog(f_band,p_n,label=r'$S_{n}$')
	plt.legend()
	plt.title('{}'.format(point_estimate))
	plt.savefig(dir+'residuals_2nd_noises_in_data_equal_arm_posteriors_{}.png'.format(point_estimate))
	plt.show()

	print('likelihood {} value EQUAL ARM'.format(point_estimate))
	#print('value')
	print(value)

	return value	

#See pages 578-589
def likelihood_analytical(x_combo_f,y_combo_f,z_combo_f,delay_array,point_estimate):

	x_x = np.real(x_combo_f)
	x_y = np.imag(x_combo_f)

	y_x = np.real(y_combo_f)
	y_y = np.imag(y_combo_f)

	z_x = np.real(z_combo_f)
	z_y = np.imag(z_combo_f)

	#covariance_here = covariance_S_n_f(f_band).T


	
	covariance_here,m,n,p,ax,ay,bx,by,cx,cy = covariance_unequal_arm(delay_array)


	#det_here = np.array(np.linalg.det(np.pi*covariance_here),dtype=np.complex128)
	log_term_factor = 3*np.log(np.pi)
	determinant = 2*(ax*bx*cx + ay*by*cx + ax*by*cy - ay*bx*cy) - p*(ax**2+ay**2) - n*(bx**2+by**2) - m*(cx**2+cy**2) + m*n*p
	log_term_determinant = np.log(determinant)
	
	xax = (x_x**2+x_y**2)*(n*p-cx**2-cy**2)
	yey = (y_x**2+y_y**2)*(m*p-bx**2-by**2)
	ziz = (z_x**2+z_y**2)*(m*n-ax**2-ay**2)

	xdy = -1*x_x*y_y*p*ay - x_x*y_y*bx*cy + x_x*y_y*by*cx - x_y*y_y*p*ax + x_y*y_y*bx*cx \
		+ x_y*y_y*by*cy - x_x*y_x*p*ax + x_x*y_x*bx*cx + x_x*y_x*by*cy + x_y*y_x*p*ay \
		+ x_y*y_x*bx*cy - x_y*y_x*by*cx

	yhz = -1*y_x*z_x*m*cx + y_x*z_x*ax*bx + y_x*z_x*ay*by + y_y*z_x*m*cy + y_y*z_x*ay*bx \
		- y_y*z_x*ax*by - y_x*z_y*m*cy - y_x*z_y*ay*bx + y_x*z_y*ax*by - y_y*z_y*m*cx \
		+ y_y*z_y*ax*bx + y_y*z_y*ay*by

	xgz = x_x*z_x*ax*cx - x_x*z_x*ay*cy - x_x*z_x*n*bx - x_y*z_x*ax*cy - x_y*z_x*ay*cx \
		+ x_y*z_x*n*by + x_x*z_y*ax*cy + x_x*z_y*ay*cx - x_x*z_y*n*by + x_y*z_y*ax*cx \
		- x_y*z_y*ay*cy - x_y*z_y*n*bx

	chi_2 = 1/determinant*(xax + 2*xdy +2*xgz +yey + 2*yhz + ziz)

	

	'''
	log_term_determinant = np.log(np.power(np.pi,3)*(2*(ax*bx*cx + ay*by*cx + ax*by*cy \
		- ay*bx*cy) - p*(ax**2+ay**2) - n*(bx**2+by**2) - m*(cx**2+cy**2) + m*n*p))
	'''

	
	#print(np.sum(log_term_determinant))
	#chi_2 = conj_x*inv_T[0][0][:]*x_combo_f + conj_y*inv_T[1][1][:]*y_combo_f + conj_z*inv_T[2][2][:]*z_combo_f + conj_x*inv_T[0][1][:]*y_combo_f + conj_y*inv_T[1][0][:]*x_combo_f + conj_y*inv_T[1][2][:]*z_combo_f + conj_z*inv_T[2][1][:]*y_combo_f + conj_x*inv_T[0][2][:]*z_combo_f + conj_z*inv_T[2][0][:]*x_combo_f

	print('log term determinant')
	print(log_term_determinant)
	print('np.sum(log_term_determinant)')

	print(np.sum(log_term_determinant))
	print('np.sum(chi_2)')
	print(np.sum(chi_2))
	#value = -1/2*(np.sum(chi_2))-np.sum(log_term_factor)
	#value = -1*np.sum(chi_2) - np.sum(log_term_determinant)
	value = -1*np.sum(chi_2) - log_term_factor - np.sum(log_term_determinant) 

	'''
	#....................................................................................
	#.....................Equal Arm Noise Covariance Matrix..............................
	#....................................................................................

	a,b_ = covariance_equal_arm(delay_array)

	#....................................................................................
	#.....................AET TDI and Noise Covariance Matrix..............................
	#....................................................................................


	#New AET from LDC Manual
	A = 1/np.sqrt(2)*(z_combo_f - x_combo_f)
	E = 1/np.sqrt(6)*(x_combo_f - 2*y_combo_f + z_combo_f)
	T = 1/np.sqrt(3)*(x_combo_f + y_combo_f + z_combo_f)

	noise_A_result = a-b_
	noise_T_result = a+2*b_

	'''
	plt.loglog(f_band,np.abs(x_combo_f)**2,label='Resiudual XX')
	plt.loglog(f_band,np.abs(x_combo_f)*np.abs(y_combo_f),label='Residual XY')
	#plt.loglog(f_band,np.abs(A)**2,label='Residual A Channel',alpha=0.5)
	#plt.loglog(f_band,np.abs(E)**2,label='Residual E Channel',alpha=0.5)
	#plt.loglog(f_band,np.abs(T)**2,label='Residual T Channel',alpha=0.5)
	plt.loglog(f_band,m,label=r'$\Sigma_{UNEQUAL}[0][0]$')
	plt.loglog(f_band,np.abs(ax),label=r'$\Sigma_{UNEQUAL}[0][1]$')
	#plt.loglog(f_band,a,label=r'$\Sigma_{EQUAL}[0][0]$')
	#plt.loglog(f_band,b_,label=r'$\Sigma_{EQUAL}[0][1]$')
	#plt.loglog(f_band,noise_A_result,label=r'$\Sigma_{AET}[0][0]$')
	#plt.loglog(f_band,noise_T_result,label=r'$\Sigma_{AET}[2][2]$')
	#plt.loglog(f_band,p_n,label=r'$S_{n}$')
	plt.legend()
	plt.title('{}'.format(point_estimate))
	plt.savefig(dir+'residuals_2nd_noises_in_data_unequal_arm_posteriors_{}.png'.format(point_estimate))
	plt.show()

	print('likelihood {} value UNEQUAL ARM'.format(point_estimate))
	#print('value')
	print(value)

	return value	








data_science_ref =  np.genfromtxt('/Users/jessica/Desktop/Project_1/Rigid_Rotation/split_interferometry/Production/Filter_Length_Tests/Data/data_fs_4_N=49.dat',names=True)

s31 = data_science_ref['s31']
s21 = data_science_ref['s21']
s32 = data_science_ref['s32']
s12 = data_science_ref['s12']
s23 = data_science_ref['s23']
s13 = data_science_ref['s13']

s31_noise = data_science_ref['s31_noise']
s21_noise = data_science_ref['s21_noise']
s32_noise = data_science_ref['s32_noise']
s12_noise = data_science_ref['s12_noise']
s23_noise = data_science_ref['s23_noise']
s13_noise = data_science_ref['s13_noise']

tau31 = data_science_ref['tau31']
tau21 = data_science_ref['tau21']
tau12 = data_science_ref['tau12']
tau32 = data_science_ref['tau32']
tau23 = data_science_ref['tau23']
tau13 = data_science_ref['tau13']

tau31_noise = data_science_ref['tau31_noise']
tau21_noise = data_science_ref['tau21_noise']
tau12_noise = data_science_ref['tau12_noise']
tau32_noise = data_science_ref['tau32_noise']
tau23_noise = data_science_ref['tau23_noise']
tau13_noise = data_science_ref['tau13_noise']

eps31 = data_science_ref['eps31']
eps21 = data_science_ref['eps21']
eps12 = data_science_ref['eps12']
eps32 = data_science_ref['eps32']
eps23 = data_science_ref['eps23']
eps13 = data_science_ref['eps13']

eps31_noise = data_science_ref['eps31_noise']
eps21_noise = data_science_ref['eps21_noise']
eps12_noise = data_science_ref['eps12_noise']
eps32_noise = data_science_ref['eps32_noise']
eps23_noise = data_science_ref['eps23_noise']
eps13_noise = data_science_ref['eps13_noise']


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

length = len(s31)

#filter length
number_n = 29

#m = 51
nearest_number = length
# number points in filter
if nearest_number%2 == 0:
	h_points = np.arange(-nearest_number/2.0,nearest_number/2.0,1)
else:
	h_points = np.arange(-(nearest_number-1)/2.0,(nearest_number-1)/2.0+1,1)


L_1_real = 8.339095192
L_1_p_real = 8.338994879
L_2_real = 8.338867041
L_2_p_real = 8.339095192
L_3_real = 8.338994879
L_3_p_real = 8.338867041


f_s = 4

f_samp = 4

time_length = 1*24*3600

#LISA sensitivity band
f_min = 1.0e-4 # (= 0.0009765625)
f_max = 1.0e-1

L_arm = 2.5e9
avg_L = L_arm/const.c.value

log_term_factor = 3*np.log(np.pi)




#data = np.recfromtxt('chain_data_overlap_add.dat',names = True)
#data = np.recfromtxt('/Users/jessica/Desktop/Project_1/Rigid_Rotation/MCMC/chainfile.dat',names = True)

data_AET_N_29 = np.recfromtxt('/Users/jessica/Desktop/Project_1/Rigid_Rotation/split_interferometry/Production/Noise_Covariance_Matrix_Tests/chainfile_noise_matrix_AET_978_29.dat',names = True)

data_equal_29 = np.recfromtxt('/Users/jessica/Desktop/Project_1/Rigid_Rotation/split_interferometry/Production/Noise_Covariance_Matrix_Tests/chainfile_noise_matrix_equal_arm_967_29.dat',names = True)

#N does not really equal 49 here, it's 29 too
data_unequal_N_49 = np.recfromtxt('/Users/jessica/Desktop/Project_1/Rigid_Rotation/split_interferometry/Production/Noise_Covariance_Matrix_Tests/chainfile_noise_matrix_unequal_arm_977_29.dat',names = True)
#data_true = np.recfromtxt('/Users/jessica/Desktop/Project_1/Rigid_Rotation/MCMC/new_simulation/include_2nd_noises/chainfile_noise_matrix_unequal_arm_793.dat',names = True)

dir = '/Users/jessica/Desktop/Project_1/Rigid_Rotation/split_interferometry/Production/Noise_Covariance_Matrix_Tests/'

likelihood_AET = data_AET_N_29['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_AET))
likelihood_AET = data_AET_N_29['likelihood'][removed::]
L_3_AET = data_AET_N_29['L_3'][removed::]
L_2_AET = data_AET_N_29['L_2'][removed::]
L_1_AET = data_AET_N_29['L_1'][removed::]
L_3_p_AET = data_AET_N_29['L_3_p'][removed::]
L_2_p_AET = data_AET_N_29['L_2_p'][removed::]
L_1_p_AET = data_AET_N_29['L_1_p'][removed::]
chi_2_AET = data_AET_N_29['sum_chi_2'][removed::]
maximum_AET = np.max(likelihood_AET)
indice_max_L_AET = np.where(likelihood_AET==maximum_AET)


likelihood_equal = data_equal_29['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_equal))
likelihood_equal = data_equal_29['likelihood'][removed::]
L_3_equal = data_equal_29['L_3'][removed::]
L_2_equal = data_equal_29['L_2'][removed::]
L_1_equal = data_equal_29['L_1'][removed::]
L_3_p_equal = data_equal_29['L_3_p'][removed::]
L_2_p_equal = data_equal_29['L_2_p'][removed::]
L_1_p_equal = data_equal_29['L_1_p'][removed::]
chi_2_equal = data_equal_29['sum_chi_2'][removed::]
maximum_equal = np.max(likelihood_equal)
indice_max_L_equal = np.where(likelihood_equal==maximum_equal)

likelihood_unequal = data_unequal_N_49['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_unequal))
likelihood_unequal = data_unequal_N_49['likelihood'][removed::]
L_3_unequal = data_unequal_N_49['L_3'][removed::]
L_2_unequal = data_unequal_N_49['L_2'][removed::]
L_1_unequal = data_unequal_N_49['L_1'][removed::]
L_3_p_unequal = data_unequal_N_49['L_3_p'][removed::]
L_2_p_unequal = data_unequal_N_49['L_2_p'][removed::]
L_1_p_unequal = data_unequal_N_49['L_1_p'][removed::]
chi_2_unequal = data_unequal_N_49['sum_chi_2'][removed::]
maximum_unequal = np.max(likelihood_unequal)
indice_max_L_unequal = np.where(likelihood_unequal==maximum_unequal)



beg_ind,end_ind = cut_data(L_3_real,L_2_real,L_1_real,L_3_p_real,L_2_p_real,L_1_p_real,f_s,length)
window = cosine(length)[beg_ind:end_ind:]
#secondary noise power for likelihood calculation
f_band = np.fft.rfftfreq(len(window),1/f_s)
#p_n = secondary_noise_power(f_band)
indices_f_band = np.where(np.logical_and(f_band>=f_min, f_band<=f_max))
f_band=f_band[indices_f_band]


Sy_PM = S_y_proof_mass_new_frac_freq(f_band)
Sy_OP = S_y_OMS_frac_freq(f_band)


x_combo_max_L_LFN_AET = x_combo_LFN(L_3_AET[indice_max_L_AET][0],L_2_AET[indice_max_L_AET][0],L_3_p_AET[indice_max_L_AET][0],L_2_p_AET[indice_max_L_AET][0])
x_combo_max_L_LFN_equal = x_combo_LFN(L_3_equal[indice_max_L_equal][0],L_2_equal[indice_max_L_equal][0],L_3_p_equal[indice_max_L_equal][0],L_2_p_equal[indice_max_L_equal][0])
x_combo_max_L_LFN_unequal = x_combo_LFN(L_3_unequal[indice_max_L_unequal][0],L_2_unequal[indice_max_L_unequal][0],L_3_p_unequal[indice_max_L_unequal][0],L_2_p_unequal[indice_max_L_unequal][0])
y_combo_max_L_LFN_unequal = x_combo_LFN(L_3_unequal[indice_max_L_unequal][0],L_1_unequal[indice_max_L_unequal][0],L_3_p_unequal[indice_max_L_unequal][0],L_1_p_unequal[indice_max_L_unequal][0])
z_combo_max_L_LFN_unequal = x_combo_LFN(L_2_unequal[indice_max_L_unequal][0],L_1_unequal[indice_max_L_unequal][0],L_2_p_unequal[indice_max_L_unequal][0],L_1_p_unequal[indice_max_L_unequal][0])

#Covariance matrices for the 3 different comparisons
a_max_L,b_max_L = covariance_equal_arm(np.array([L_1_equal[indice_max_L_equal][0],L_1_p_equal[indice_max_L_equal][0],L_2_equal[indice_max_L_equal][0],L_2_p_equal[indice_max_L_equal][0],L_3_equal[indice_max_L_equal][0],L_3_p_equal[indice_max_L_equal][0]]))
m,n,p,a_x,ay,bx,by,cx,cy = covariance_unequal_arm(np.array([L_1_unequal[indice_max_L_unequal][0],L_1_p_unequal[indice_max_L_unequal][0],L_2_unequal[indice_max_L_unequal][0],L_2_p_unequal[indice_max_L_unequal][0],L_3_unequal[indice_max_L_unequal][0],L_3_p_unequal[indice_max_L_unequal][0]]))
a,b_ = covariance_equal_arm(np.array([L_1_AET[indice_max_L_AET][0],L_1_p_AET[indice_max_L_AET][0],L_2_AET[indice_max_L_AET][0],L_2_p_AET[indice_max_L_AET][0],L_3_AET[indice_max_L_AET][0],L_3_p_AET[indice_max_L_AET][0]]))
noise_A_result = (a-b_)
noise_T_result = (a+2*b_)

#named this because it's the test described on page 191 in notebook
#dir = '/Users/jessica/Desktop/Project_1/Rigid_Rotation/plots/test_223_a/'

#true_diff = L_1_real-L_2_real



L_3_here_unequal = (L_3_unequal-L_3_real)*1e9
L_2_here_unequal = (L_2_unequal-L_2_real)*1e9
L_1_here_unequal = (L_1_unequal-L_1_real)*1e9
L_3_p_here_unequal = (L_3_p_unequal-L_3_p_real)*1e9
L_2_p_here_unequal = (L_2_p_unequal-L_2_p_real)*1e9
L_1_p_here_unequal = (L_1_p_unequal-L_1_p_real)*1e9


L_3_here_equal = (L_3_equal-L_3_real)*1e9
L_2_here_equal = (L_2_equal-L_2_real)*1e9
L_1_here_equal = (L_1_equal-L_1_real)*1e9
L_3_p_here_equal = (L_3_p_equal-L_3_p_real)*1e9
L_2_p_here_equal = (L_2_p_equal-L_2_p_real)*1e9
L_1_p_here_equal = (L_1_p_equal-L_1_p_real)*1e9

L_3_here_AET = (L_3_AET-L_3_real)*1e9
L_2_here_AET = (L_2_AET-L_2_real)*1e9
L_1_here_AET = (L_1_AET-L_1_real)*1e9
L_3_p_here_AET = (L_3_p_AET-L_3_p_real)*1e9
L_2_p_here_AET = (L_2_p_AET-L_2_p_real)*1e9
L_1_p_here_AET = (L_1_p_AET-L_1_p_real)*1e9

L_3_here_unequal_median = np.median(L_3_here_unequal)
L_2_here_unequal_median = np.median(L_2_here_unequal)
L_1_here_unequal_median = np.median(L_1_here_unequal)
L_3_p_here_unequal_median = np.median(L_3_p_here_unequal)
L_2_p_here_unequal_median = np.median(L_2_p_here_unequal)
L_1_p_here_unequal_median = np.median(L_1_p_here_unequal)

L_3_here_unequal_lower_90 = np.quantile(L_3_here_unequal,0.05)
L_2_here_unequal_lower_90 = np.quantile(L_2_here_unequal,0.05)
L_1_here_unequal_lower_90 = np.quantile(L_1_here_unequal,0.05)
L_3_p_here_unequal_lower_90 = np.quantile(L_3_p_here_unequal,0.05)
L_2_p_here_unequal_lower_90 = np.quantile(L_2_p_here_unequal,0.05)
L_1_p_here_unequal_lower_90 = np.quantile(L_1_p_here_unequal,0.05)

L_3_here_unequal_upper_90 = np.quantile(L_3_here_unequal,0.95)
L_2_here_unequal_upper_90 = np.quantile(L_2_here_unequal,0.95)
L_1_here_unequal_upper_90 = np.quantile(L_1_here_unequal,0.95)
L_3_p_here_unequal_upper_90 = np.quantile(L_3_p_here_unequal,0.95)
L_2_p_here_unequal_upper_90 = np.quantile(L_2_p_here_unequal,0.95)
L_1_p_here_unequal_upper_90 = np.quantile(L_1_p_here_unequal,0.95)


L_3_here_equal_median = np.median(L_3_here_equal)
L_2_here_equal_median = np.median(L_2_here_equal)
L_1_here_equal_median = np.median(L_1_here_equal)
L_3_p_here_equal_median = np.median(L_3_p_here_equal)
L_2_p_here_equal_median = np.median(L_2_p_here_equal)
L_1_p_here_equal_median = np.median(L_1_p_here_equal)

L_3_here_equal_lower_90 = np.quantile(L_3_here_equal,0.05)
L_2_here_equal_lower_90 = np.quantile(L_2_here_equal,0.05)
L_1_here_equal_lower_90 = np.quantile(L_1_here_equal,0.05)
L_3_p_here_equal_lower_90 = np.quantile(L_3_p_here_equal,0.05)
L_2_p_here_equal_lower_90 = np.quantile(L_2_p_here_equal,0.05)
L_1_p_here_equal_lower_90 = np.quantile(L_1_p_here_equal,0.05)

L_3_here_equal_upper_90 = np.quantile(L_3_here_equal,0.95)
L_2_here_equal_upper_90 = np.quantile(L_2_here_equal,0.95)
L_1_here_equal_upper_90 = np.quantile(L_1_here_equal,0.95)
L_3_p_here_equal_upper_90 = np.quantile(L_3_p_here_equal,0.95)
L_2_p_here_equal_upper_90 = np.quantile(L_2_p_here_equal,0.95)
L_1_p_here_equal_upper_90 = np.quantile(L_1_p_here_equal,0.95)


L_3_here_AET_median = np.median(L_3_here_AET)
L_2_here_AET_median = np.median(L_2_here_AET)
L_1_here_AET_median = np.median(L_1_here_AET)
L_3_p_here_AET_median = np.median(L_3_p_here_AET)
L_2_p_here_AET_median = np.median(L_2_p_here_AET)
L_1_p_here_AET_median = np.median(L_1_p_here_AET)

L_3_here_AET_lower_90 = np.quantile(L_3_here_AET,0.05)
L_2_here_AET_lower_90 = np.quantile(L_2_here_AET,0.05)
L_1_here_AET_lower_90 = np.quantile(L_1_here_AET,0.05)
L_3_p_here_AET_lower_90 = np.quantile(L_3_p_here_AET,0.05)
L_2_p_here_AET_lower_90 = np.quantile(L_2_p_here_AET,0.05)
L_1_p_here_AET_lower_90 = np.quantile(L_1_p_here_AET,0.05)

L_3_here_AET_upper_90 = np.quantile(L_3_here_AET,0.95)
L_2_here_AET_upper_90 = np.quantile(L_2_here_AET,0.95)
L_1_here_AET_upper_90 = np.quantile(L_1_here_AET,0.95)
L_3_p_here_AET_upper_90 = np.quantile(L_3_p_here_AET,0.95)
L_2_p_here_AET_upper_90 = np.quantile(L_2_p_here_AET,0.95)
L_1_p_here_AET_upper_90 = np.quantile(L_1_p_here_AET,0.95)

L_3_median = [L_3_here_AET_median,L_3_here_equal_median,L_3_here_unequal_median]
L_2_median = [L_2_here_AET_median,L_2_here_equal_median,L_2_here_unequal_median]
L_1_median = [L_1_here_AET_median,L_1_here_equal_median,L_1_here_unequal_median]
L_3_p_median = [L_3_p_here_AET_median,L_3_p_here_equal_median,L_3_p_here_unequal_median]
L_2_p_median = [L_2_p_here_AET_median,L_2_p_here_equal_median,L_2_p_here_unequal_median]
L_1_p_median = [L_1_p_here_AET_median,L_1_p_here_equal_median,L_1_p_here_unequal_median]

L_3_upper = [L_3_here_AET_upper_90,L_3_here_equal_upper_90,L_3_here_unequal_upper_90]
L_2_upper = [L_2_here_AET_upper_90,L_2_here_equal_upper_90,L_2_here_unequal_upper_90]
L_1_upper = [L_1_here_AET_upper_90,L_1_here_equal_upper_90,L_1_here_unequal_upper_90]
L_3_p_upper = [L_3_p_here_AET_upper_90,L_3_p_here_equal_upper_90,L_3_p_here_unequal_upper_90]
L_2_p_upper = [L_2_p_here_AET_upper_90,L_2_p_here_equal_upper_90,L_2_p_here_unequal_upper_90]
L_1_p_upper = [L_1_p_here_AET_upper_90,L_1_p_here_equal_upper_90,L_1_p_here_unequal_upper_90]

L_3_lower = [L_3_here_AET_lower_90,L_3_here_equal_lower_90,L_3_here_unequal_lower_90]
L_2_lower = [L_2_here_AET_lower_90,L_2_here_equal_lower_90,L_2_here_unequal_lower_90]
L_1_lower = [L_1_here_AET_lower_90,L_1_here_equal_lower_90,L_1_here_unequal_lower_90]
L_3_p_lower = [L_3_p_here_AET_lower_90,L_3_p_here_equal_lower_90,L_3_p_here_unequal_lower_90]
L_2_p_lower = [L_2_p_here_AET_lower_90,L_2_p_here_equal_lower_90,L_2_p_here_unequal_lower_90]
L_1_p_lower = [L_1_p_here_AET_lower_90,L_1_p_here_equal_lower_90,L_1_p_here_unequal_lower_90]

print('L_3_median')
print(L_3_median)

np.savetxt('manual_credible_intervals.txt',np.c_[L_3_median,L_3_upper,L_3_lower,L_2_median,L_2_upper,L_2_lower,L_1_median,L_1_upper,L_1_lower,L_3_p_median,L_3_p_upper,L_3_p_lower,L_2_p_median,L_2_p_upper,L_2_p_lower,L_1_p_median,L_1_p_upper,L_1_p_lower])
data_plot_unequal = np.array([L_3_here_unequal,L_2_here_unequal,L_1_here_unequal,L_3_p_here_unequal,L_2_p_here_unequal,L_1_p_here_unequal]).T

data_plot_equal = np.array([L_3_here_equal,L_2_here_equal,L_1_here_equal,L_3_p_here_equal,L_2_p_here_equal,L_1_p_here_equal]).T

data_plot_AET = np.array([L_3_here_AET,L_2_here_AET,L_1_here_AET,L_3_p_here_AET,L_2_p_here_AET,L_1_p_here_AET]).T





#----------------------------------------------------------------------------
#chain consumer method
#----------------------------------------------------------------------------

c = ChainConsumer()
parameters=[ r"$\delta L_{3} \: (ns)$", r"$\delta L_{2} \: (ns)$",r"$\delta L_{1} \: (ns)$", r"$\delta L^{'}_{3} \: (ns)$",r"$\delta L^{'}_{2} \: (ns)$",r"$\delta L^{'}_{1} \: (ns)$"]

#parameters=[ r"$L_{3} \: (ns)$", r"$L_{2} \: (ns)$",r"$L_{1} \: (ns)$", r"$L^{'}_{3} \: (ns)$",r"$L^{'}_{2} \: (ns)$",r"$L^{'}_{1} \: (ns)$"]
#parameters=[ r"$L_{3}-L_{3_{True}}$", r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"]
c.add_chain(data_plot_AET, parameters=parameters,color=color.to_hex('purple'), name = 'AET')
c.add_chain(data_plot_equal, parameters=parameters, color=color.to_hex('blue'), name = 'Equal-arm XYZ')
c.add_chain(data_plot_unequal, parameters=parameters, color=color.to_hex('green'), name = 'Unequal-arm XYZ')


#90% credible interval, kde=False means no parameter smoothing
c.configure(sigmas=[0,1.645],kde=False,smooth=1,summary=True,legend_kwargs={'fontsize':'x-large','bbox_to_anchor':(-2, 0.9)})
#c.configure(sigmas=[0,1.645])
#c.plotter.plot_summary(chains=['LaGrange N=59','LaGrange N=49','LaGrange N=43','LaGrange N=41','LaGrange N=37','LaGrange N=35','LaGrange N=33'],filename='summary_plot_chain_consumer_N=43_vs_41_vs_37_vs_35_vs_33_vs_49_vs_59.png',display=True, truth=[0e1,0e1,0e1,0e1,0e1,0e1])
#c.plotter.plot(chains=['LaGrange N=59','LaGrange N=49','LaGrange N=43','LaGrange N=41','LaGrange N=37','LaGrange N=35','LaGrange N=33','LaGrange N=31','LaGrange N=29','LaGrange N=27','LaGrange N=25','LaGrange N=21'],filename='comparison_chain_consumer_N=43_vs_41_vs_37_vs_35_vs_33_vs_49_vs_59_vs 31_vs_29_vs_27_vs_25_vs_21.png',display=True, truth=[0e1,0e1,0e1,0e1,0e1,0e1], legend=True)
c.plotter.plot(chains=['AET','Equal-arm XYZ','Unequal-arm XYZ'], figsize=(7,7),truth=[0e1,0e1,0e1,0e1,0e1,0e1], display=True, filename='chain_consumer_equal_vs_unequal_vs_AET.png', legend=True)
#fig = c.plotter.plot(chains=['AET','Equal-arm XYZ','Unequal-arm XYZ'], truth=[0e1,0e1,0e1,0e1,0e1,0e1], display=True, legend=True)

c.analysis.get_latex_table(parameters=parameters, hlines=True, blank_fill='--', filename='noise_cov_matrix_table')



'''
gs = fig.axes[9].get_gridspec()
#gs = fig.axes.get_gridspec()


# remove the underlying axes
for ax in fig.axes[10:12]:
#for ax in fig.axes[1:, -1]:
    ax.remove()
axbig = fig.add_subplot(gs[10:12])
#axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5), xycoords='axes fraction', va='center')
plt.sca(axbig)

'''


'''
plt.hist(chi_2_AET,density=True,alpha=0.5,color='purple',label='AET')
plt.hist(chi_2_equal,density=True,alpha=0.5,color='blue',label='Equal-Arm')
plt.hist(chi_2_unequal,density=True,alpha=0.5,color='green',label='Unequal-Arm')
plt.xlabel(r'$\sum\limits^{f_{max}}_{i=f_{min}} \chi^{2}_{i}$')
plt.savefig('chi_2_posteriors_combined.png')
plt.show()
plt.close()
'''

plt.loglog(np.fft.rfftfreq(len(s31[beg_ind:end_ind:]),1/f_s)[indices_f_band],np.power(np.abs(np.fft.rfft(s31[beg_ind:end_ind:],norm='ortho'))[indices_f_band],2),color='k',label=r'$s_{31}$')
plt.loglog(f_band,np.abs(x_combo_max_L_LFN_AET)**2,alpha=0.5,color='purple',label='AET LFN')
plt.loglog(f_band,np.abs(x_combo_max_L_LFN_equal)**2,alpha=0.5,color='blue',label='Equal-arm LFN')
plt.loglog(f_band,np.abs(x_combo_max_L_LFN_unequal)**2,alpha=0.5,color='green',label='Unequal-arm LFN')
plt.loglog(f_band,noise_A_result,linestyle ='--',label=r'$C_{AET}[0][0]$')
plt.loglog(f_band,noise_T_result,label=r'$C_{AET}[2][2]$')
plt.loglog(f_band,a_max_L,label=r'$C_{EQUAL}[0][0]$')
plt.loglog(f_band,b_max_L,label=r'$C_{EQUAL}[0][1]$')
plt.loglog(f_band,m,linestyle='--',label=r'$C_{UNEQUAL}[0][0]$')
plt.loglog(f_band,np.abs(a_x),linestyle='--',label=r'$C_{UNEQUAL}[0][1]$')
#plt.legend(loc='center left',bbox_to_anchor=(1.001, 0.5,0.15,0.5),fontsize='x-small')
plt.legend(loc='center left',bbox_to_anchor=(0, 0.5),fontsize='x-small',ncol=2)
plt.title(r'Max $\mathcal{L}$ Parameters X Channel')
plt.xlabel('f [Hz]',fontsize=12)
plt.ylabel(r'PSD [$Hz^{-1}$]',fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('residuals_XX_separated_max_L_noise_cov_test.png')
plt.show()




'''
c_chi_2 = ChainConsumer()
parameters=['59','49','41','43','37','35','33','31','29','27','25','21']
c_chi_2.add_chain(chi_2_data_plot,parameters=parameters,name='chi_2')
c_chi_2.plotter.plot_distributions(parameters=parameters, truth=None, extents=None, display=True, filename='chi_2.png', chains='chi_2', col_wrap=4, figsize=None, blind=None, log_scales=None)
'''
#----------------------------------------------------------------------------
#corner py method
#----------------------------------------------------------------------------
'''
green_line = mlines.Line2D([], [], color='green', label='LaGrange N=41')
blue_line = mlines.Line2D([], [], color='blue', label='LaGrange N=43')
purple_line = mlines.Line2D([], [], color='purple', label = 'LaGrange N=37')
orange_line = mlines.Line2D([], [], color='orange', label = 'LaGrange N=35')
red_line = mlines.Line2D([], [], color='red', label = 'LaGrange N=33')
magenta_line = mlines.Line2D([], [], color='magenta', label = 'LaGrange N=49')
cyan_line = mlines.Line2D([], [], color='cyan', label = 'LaGrange N=59')

#90% credible interval
fig = corner.corner(data_plot_41,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.05, 0.95],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='green')
corner.corner(data_plot_43,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.05, 0.95],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='blue',fig=fig)

corner.corner(data_plot_37,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.05, 0.95],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='purple',fig=fig)
corner.corner(data_plot_35,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.05, 0.95],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='orange',fig=fig)
corner.corner(data_plot_33,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.05, 0.95],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='red',fig=fig)
corner.corner(data_plot_49,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.05, 0.95],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='magenta',fig=fig)
corner.corner(data_plot_59,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.05, 0.95],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='cyan',fig=fig)
plt.legend(handles=[blue_line,green_line,purple_line,orange_line,red_line,magenta_line,cyan_line], bbox_to_anchor=(0., 1.0, 1., .0), loc=4)
#plt.legend(handles=[blue_line,green_line,purple_line], bbox_to_anchor=(0., 1.0, 1., .0), loc=4)
plt.savefig(dir+'corner_plot_N=43_vs_41_vs_37_vs_35_vs_33_vs_49_vs_59_90_CI.png')
plt.show()
'''

#----------------------------------------------------------------------------
#chi^2 histograms
#----------------------------------------------------------------------------
'''
sixteen_old = np.quantile(chi_2_43,0.16)
sixteen_old_also = np.quantile(chi_2_37,0.16)
sixteen_new = np.quantile(chi_2_41,0.16)

eightyfour_old_also =  np.quantile(chi_2_37,0.84)
eightyfour_old =  np.quantile(chi_2_43,0.84)
eightyfour_new =  np.quantile(chi_2_41,0.84)
'''

'''
plt.hist(chi_2_59,label='N=59',color='cyan')
plt.hist(chi_2_49,label='N=49',alpha=0.5,color='magenta')
plt.hist(chi_2_43,label='N=43',alpha=0.5,color='blue')
plt.hist(chi_2_41,label='N=41',alpha=0.5,color='green')
plt.hist(chi_2_37,label='N=37',alpha=0.5,color='purple')
plt.hist(chi_2_35,label='N=35',alpha=0.5,color='orange')
plt.hist(chi_2_33,label='N=33',alpha=0.5,color='red')
plt.hist(chi_2_31,label='N=31',alpha=0.5,color='red')
plt.hist(chi_2_29,label='N=29',alpha=0.5,color='green')
plt.hist(chi_2_27,label='N=27',alpha=0.5,color='brown')
plt.hist(chi_2_25,label='N=25',alpha=0.5,color='black')
#plt.hist(chi_2_23,label='N=23',alpha=0.5)
plt.hist(chi_2_21,label='N=21',alpha=0.5,color='orange')

plt.legend()
plt.title(r'$\Sigma \chi^{2}$')
plt.show()
'''

'''
plt.hist(chi_2_41,color='green',label='LaGrange N=49')
plt.hist(chi_2_43,color='blue',label='LaGrange N=43')
#plt.hist(chi_2_37,color='purple',label='LaGrange N=59')
#plt.axvline(sixteen_old_also, color='k', linestyle='dashed')
plt.axvline(sixteen_old, color='k', linestyle='dashed')
plt.axvline(sixteen_new, color='k', linestyle='dashed')
#plt.axvline(eightyfour_old_also, color='k', linestyle='dashed')
plt.axvline(eightyfour_old, color='k', linestyle='dashed')
plt.axvline(eightyfour_new, color='k', linestyle='dashed')
plt.legend()
plt.title(r'$\Sigma \chi^{2}$')
plt.savefig(dir+'chi_2_compare_filter_lengths_49_43.png')
plt.show()
'''



'''
#1 sigma credible interval
fig = corner.corner(data_plot_1,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.16, 0.84],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='green')
corner.corner(data_plot_2,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.16, 0.84],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
					use_math_text=True,color='blue',fig=fig)
corner.corner(data_plot_3,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.16, 0.84],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
					use_math_text=True,color='purple',fig=fig)
plt.legend(handles=[blue_line,green_line,purple_line], bbox_to_anchor=(0., 1.0, 1., .0), loc=4)
plt.savefig(dir+'corner_plot_compare_filter_lengths_39_49_59.png')
plt.show()
'''


'''
corner.corner(data_plot_1_true,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.16, 0.84],show_titles=False, title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='purple',fig=fig)
'''


'''
plt.hist(L_3_here,bins=20,density=True,label='X Channel')
plt.hist(L_3_here_43,bins=20,density=True,label='XYZ Channels',alpha=0.5)
plt.xlabel(r"$L_{3}-L_{3_{True}}$")
#plt.title('X Channel Only')
plt.legend()
plt.show()

plt.hist(L_2_here,bins=20,density=True,label='X Channel')
plt.hist(L_2_here_43,bins=20,density=True,label='XYZ Channels',alpha=0.5)
plt.xlabel(r"$L_{2}-L_{2_{True}}$")
#plt.title('X Channel Only')
plt.legend()
plt.show()
'''



'''
data_estimation = np.array([L_3,L_2,L_3_p,L_2_p])
print('likelihood')
print(likelihood)
#data_plot_1 = np.vstack([likelihood,L_3,L_2,L_3_p,L_2_p])
print('data_plot_1')
print(data_plot_1)
print('data_estimation')
print(data_estimation)
cc_L_3_L_3_p = np.corrcoef(L_3,L_3_p)
cc_L_3_L_2_p = np.corrcoef(L_3,L_2_p)
cc_L_3_L_2 = np.corrcoef(L_3,L_2)
cc_L_3_p_L_2 = np.corrcoef(L_3_p,L_2)
cc_L_3_p_L_2_p = np.corrcoef(L_3_p,L_2_p)
cc_L_2_p_L_2 = np.corrcoef(L_2_p,L_2)
print('cc_L_3_L_3_p')
print(cc_L_3_L_3_p)
print('cc_L_3_L_2_p')
print(cc_L_3_L_2_p)
print('cc_L_3_L_2')
print(cc_L_3_L_2)
print('cc_L_3_p_L_2')
print(cc_L_3_p_L_2)
print('cc_L_3_p_L_2_p')
print(cc_L_3_p_L_2_p)
print('cc_L_2_p_L_2')
print(cc_L_2_p_L_2)

cov = np.cov(data_estimation)
print('covariance matrix')
print(cov)
corr = np.corrcoef(data_estimation)
print('normalized covariance matrix')
print(corr)



diff_L_3_L_2 = L_3-L_2
plt.hist(diff_L_3_L_2,bins = 50)
plt.title('actual_diff = {0}'.format(L_3_real-L_2_real))
plt.axvline(np.median(diff_L_3_L_2),label = 'median={0}'.format(np.median(diff_L_3_L_2)))
plt.legend()
plt.xlabel(r"$L_{3}-L_{2}$")
plt.savefig(dir+'L_3-L_2.png')
plt.show()

diff_L_3_L_2_p = L_3-L_2_p
plt.hist(diff_L_3_L_2_p,bins = 50)
plt.title('actual_diff = {0}'.format(L_3_real-L_2_p_real))
plt.axvline(np.median(diff_L_3_L_2_p),label = 'median={0}'.format(np.median(diff_L_3_L_2_p)))
plt.legend()
plt.xlabel(r"$L_{3}-L^{'}_{2}$")
plt.savefig(dir+'L_3-L_2_p.png')
plt.show()

diff_L_3_L_3_p = L_3-L_3_p
plt.hist(diff_L_3_L_3_p,bins = 50)
plt.title('actual_diff = {0}'.format(L_3_real-L_3_p_real))
plt.axvline(np.median(diff_L_3_L_3_p),label = 'median={0}'.format(np.median(diff_L_3_L_3_p)))
plt.legend()
plt.xlabel(r"$L_{3}-L^{'}_{3}$")
plt.savefig(dir+'L_3-L_3_p.png')
plt.show()

diff_L_2_L_2_p = L_2-L_2_p
plt.hist(diff_L_2_L_2_p,bins = 50)
plt.title('actual_diff = {0}'.format(L_2_real-L_2_p_real))
plt.axvline(np.median(diff_L_2_L_2_p),label = 'median={0}'.format(np.median(diff_L_2_L_2_p)))
plt.legend()
plt.xlabel(r"$L_{2}-L^{'}_{2}$")
plt.savefig(dir+'L_2-L_2_p.png')
plt.show()

diff_L_2_L_3_p = L_2-L_3_p
plt.hist(diff_L_2_L_3_p,bins = 50)
plt.title('actual_diff = {0}'.format(L_2_real-L_3_p_real))
plt.axvline(np.median(diff_L_2_L_3_p),label = 'median={0}'.format(np.median(diff_L_2_L_3_p)))
plt.legend()
plt.xlabel(r"$L_{2}-L^{'}_{3}$")
plt.savefig(dir+'L_2-L_3_p.png')
plt.show()

diff_L_3_p_L_2_p = L_3_p-L_2_p
plt.hist(diff_L_3_p_L_2_p,bins = 50)
plt.title('actual_diff = {0}'.format(L_3_p_real-L_2_p_real))
plt.axvline(np.median(diff_L_3_p_L_2_p),label = 'median={0}'.format(np.median(diff_L_3_p_L_2_p)))
plt.legend()
plt.xlabel(r"$L^{'}_{3}-L^{'}_{2}$")
plt.savefig(dir+'L_3_p-L_2_p.png')
plt.show()
'''


'''


# Set up the parameters of the problem.
ndim, nsamples = 3, 50000

# Generate some fake data.
np.random.seed(42)
data1 = np.random.randn(ndim * 4 * nsamples // 5).reshape([4 * nsamples // 5, ndim])
data2 = (4*np.random.rand(ndim)[None, :] + np.random.randn(ndim * nsamples // 5).reshape([nsamples // 5, ndim]))
data = np.vstack([data1, data2])

print('data1')
print(data1)
print('data2')
print(data2)
print('data')
print(data)
# Plot it.
figure = corner.corner(data, labels=[r"$x$", r"$y$", r"$\log \alpha$", r"$\Gamma \, [\mathrm{parsec}]$"],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})
plt.show()
'''
'''
removed = int(0.25*len(L_2))
#difference = L_1-L_2


index = np.argmax(likelihood)

print('L_3 max likelihood')
print(L_3[index])
print('L_2 max likelihood')
print(L_2[index])

plt.plot(likelihood)
plt.xlabel('iteration #')
plt.ylabel(r'$\log{\mathcal{L}}$')
plt.savefig(dir+'likelihood.png')
plt.show()

plt.plot(L_3-L_3_real)
plt.title('L_3 chain')
plt.ylabel(r'$L_{3}-L_{3_{True}}$')
plt.savefig(dir+'L_3_diff.png')
plt.show()

#plt.plot(L_2[int(len(L_1)/2)::])
plt.plot(L_2-L_2_real)
#plt.legend(['L_1','L_2'],loc='best')
plt.title('L_2 chain')
plt.ylabel(r'$L_{2}-L_{2_{True}}$')
plt.savefig(dir+'L_2_diff.png')
plt.show()

plt.plot(L_3_p-L_3_p_real)
plt.title('L_3_p chain')
plt.ylabel(r'$L_{3p}-L_{3p_{True}}$')
plt.savefig(dir+'L_3_p_diff.png')
plt.show()

#plt.plot(L_2[int(len(L_1)/2)::])
plt.plot(L_2_p-L_2_p_real)
#plt.legend(['L_1','L_2'],loc='best')
plt.title('L_2_p chain')
plt.ylabel(r'$L_{2p}-L_{2p_{True}}$')
plt.savefig(dir+'L_2_p_diff.png')
plt.show()
'''

'''

plt.plot(L_1-L_2-true_diff)
plt.ylabel(r'$\Delta L - \Delta L_{True}$')
plt.savefig(dir+'L_1_L_2_diff.png')
plt.show()

array_to_hist = difference-true_diff

weights = np.empty_like(array_to_hist)
bin =30
weights.fill(bin / (array_to_hist.max()-array_to_hist.min()) / array_to_hist.size)
plt.hist(array_to_hist, bins=bin, weights=weights)
#n,bins,patches=plt.hist(array_to_hist,density=True)
plt.xlabel(r'$\Delta L-\Delta L_{True}$ [s]')
plt.ylabel('POSTERIOR')
plt.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
#plt.xlabel(r'$L_{2}-L2_{True}$')
plt.savefig(dir+'delta_L_pdf.png')
plt.show()
'''


'''
cm = plt.cm.get_cmap('RdYlBu')
#sc = plt.scatter(L_1[removed:-1:1], L_2[removed:-1:1], c=likelihood[removed:-1:1], s=5, cmap=cm,label='likelihood')
sc = plt.scatter(L_3_p-L_3_p_real, L_2-L_2_real, c=likelihood, s=5, cmap=cm,label='likelihood')

#plt.plot(L_1,L_2)
plt.colorbar(sc)
plt.xlabel(r'$L_{3p}-L_{3p_{True}}$')
plt.ylabel(r'$L_{2}-L_{2_{True}}$')

#plt.xlim(-2e-5,2.5e-5)
#plt.ylim(-2e-5,2e-5)

#plt.ticklabel_format(axis='both', style='sci',scilimits=(-6,6))
plt.savefig(dir+'Two_D_Likelihood_L_3_p_L_2.png')
plt.show()
'''

'''
cm = plt.cm.get_cmap('RdYlBu')
#sc = plt.scatter(L_1[removed:-1:1], L_2[removed:-1:1], c=likelihood[removed:-1:1], s=5, cmap=cm,label='likelihood')
sc = plt.scatter(L_3-L_3_real, difference-true_diff, c=likelihood, s=5, cmap=cm,label='likelihood')
#plt.plot(L_1,L_2)
cbar = plt.colorbar(sc)
cbar.set_label('$log{\mathcal{L}}$')
plt.xlabel(r'$L_{1}-L_{1_{True}}$ [s]')
plt.ylabel(r'$\Delta L-\Delta L_{True}$ [s]')
plt.ticklabel_format(axis='both', style='sci',scilimits=(-6,6))
plt.savefig(dir+'Two_D_Difference_Likelihood.png')
plt.show()
'''

'''
print('L 1 median')
print(median_1)

print('L 2 median')
print(median_2)

print('L 1 lower')
print(p_l_1)
print('L 2 lower')
print(p_l_2)
print('L 1 upper')
print(p_u_1)
print('L 2 upper')
print(p_u_2)

print('max likelihood index')
print(index)
print('L_1 max likelihood')
print(L_1[index]+L_1_real)
print('L_2 max likelihood')
print(L_2[index]+L_2_real)
'''



'''
fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot3D(L_1,L_2,likelihood)
#ax2.scatter3D(0,0,np.max(likelihood),color='black')
ax2.view_init(azim=50, elev=5)
ax2.set_xlabel(r'x where $x = 8.4 \pm \frac{x}{c}$')
ax2.set_ylabel(r'x where $x = -10.4 \pm \frac{x}{c}$')
ax2.set_zlabel('likelihood')
plt.show()
'''