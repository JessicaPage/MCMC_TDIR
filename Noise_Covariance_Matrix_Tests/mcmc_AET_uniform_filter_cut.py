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


#...........................CREATING FILTERS.......................................
def filters_lagrange(delay):

	global D
	D = delay*f_samp
	global d_frac
	i, d_frac = divmod(D,1)

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

def noise_A(f):

	
	return 16.0*S_proof_mass*np.power(np.sin(np.pi*f_band*L_arm),2)*(3+2*np.cos(2.0*np.pi*f_band*L_arm)+np.cos(4.0*np.pi*f_band*L_arm))+8.0*S_optical_path*np.power(np.sin(np.pi*f_band*L_arm),2)*(2.0+np.cos(2.0*np.pi*f_band*L_arm))

def noise_T(f):


	return 2.0*(1.0+2.0*np.power(np.cos(2.0*np.pi*f_band*L_arm),2))*(4.0*np.power(np.sin(np.pi*f_band*L_arm),2)*S_proof_mass+S_optical_path)

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

	beg_ind = filter_cut+D_3_val+D_2_val+D_1_val+D_3_p_val+D_2_p_val+D_1_p_val
	end_ind = int(length-filter_cut-1)


	return beg_ind, end_ind

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

	x_combo_val = x_combo_val[beg_ind:end_ind:]

	x_combo_f_domain = np.fft.rfft(window*x_combo_val,norm='ortho')[indices_f_band]


	

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


def likelihood_AET(x_combo_f,y_combo_f,z_combo_f):



	#New AET from LDC Manual
	A = 1/np.sqrt(2)*(z_combo_f - x_combo_f)
	E = 1/np.sqrt(6)*(x_combo_f - 2*y_combo_f + z_combo_f)
	T = 1/np.sqrt(3)*(x_combo_f + y_combo_f + z_combo_f)

	Ax = np.real(A)
	Ay = np.imag(A)
	Ex = np.real(E)
	Ey = np.imag(E)
	Tx = np.real(T)
	Ty = np.imag(T)

	chi_2 = (Ax**2+Ay**2)/noise_A_result + (Ex**2+Ey**2)/noise_A_result + (Tx**2+Ty**2)/noise_T_result

	value = -1*np.sum(log_term_determinant) -np.sum(chi_2) - log_term_factor 



	return value,np.sum(log_term_determinant),np.sum(chi_2)


#........................................................................................
#...........................MCMC Functions.......................................
#........................................................................................
def proposal(mean, draw):

	
	if draw == 0:

		new_val = np.random.normal(mean,(100000/const.c.value))
		#new_val = np.random.normal(mean,(1000/const.c.value))

		return new_val
	
	elif draw == 1:

		new_val = np.random.normal(mean,(1000/const.c.value))
		return new_val
	
	elif draw == 2:

		new_val = np.random.normal(mean,(100/const.c.value))
		return new_val
	
	elif draw == 3:

		new_val = np.random.normal(mean,(1/const.c.value))
		return new_val
	



def prior(val):
	val = np.array(val)
	if (val >= low).all() and (val <= high).all():
		return 1
	else:
		return 0

	
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

time_length = 24*3600

#lagrange filter length
number_n = 29

f_min = 1e-4 # (= 0.0009765625)
f_max = 1e-1



#............................................................
#................................... DATA . ....................................
#............................................................

data =  np.genfromtxt('./../Data_Simulation/data_fs_4_N=49_FOR_RUN.dat',names=True)



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


#............................................................
#creating delayed sij and delayed sij tij combos by filtering with scipy's fftconvolve
#using equation 16 in reference_TDIR_X_1.pdf paper or page 367 in notes
#............................................................

m = 51
nearest_number = length
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

f_transfer = 3e8/(2*math.pi*L_arm)

beg_ind,end_ind = cut_data(L_3,L_2,L_1,L_3_p,L_2_p,L_1_p,f_s,length)
window = cosine(length)[beg_ind:end_ind:]
cut_data_length = len(window)
#secondary noise power for likelihood calculation
f_band = np.fft.rfftfreq(len(window),1/f_s)
#p_n = secondary_noise_power(f_band)
indices_f_band = np.where(np.logical_and(f_band>=f_min, f_band<=f_max))
f_band=f_band[indices_f_band]

#new noise PSDs afetr beginning split int implementation FRACTIONAL FREQUENCY PSD
Sy_PM = S_y_proof_mass_new_frac_freq(f_band)
Sy_OP = S_y_OMS_frac_freq(f_band)

a,b_ = covariance_equal_arm()

noise_A_result = a-b_
noise_T_result = a+2*b_

log_term_factor = 3*np.log(np.pi)
determinant  = noise_A_result*noise_A_result*noise_T_result
log_term_determinant = np.log(determinant)

#........................................................................................
#...........................MCMC Portion.......................................
#........................................................................................



#time mcmc computation time
start_time = time.time()
#for number chain iterations
#number_chain = 50000
#number_chain = 1000000
number_chain = 50000
#number_chain  = 5000
checkfile = open('chainfile_noise_matrix_AET_978_29.dat','w')
checkfile.write("#likelihood" + " " + "log_term_determ" + " " + "sum_chi_2" + " " + "L_1" + " " + "L_1_p" + " " + "L_2" + " " + "L_2_p" + " " + "L_3" + " " + "L_3_p" + " " + "Current_AR" + "\n") 
k=1


'''
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

#initial delays accepted into the chain
accept = 1
x_combo_initial = x_combo(initial_L_3,initial_L_2,initial_L_3_p, initial_L_2_p)
y_combo_initial = y_combo(initial_L_3,initial_L_1,initial_L_3_p,initial_L_1_p)
z_combo_initial = z_combo(initial_L_2,initial_L_1,initial_L_2_p,initial_L_1_p)






old_likelihood,determ_here,chi_2_here = likelihood_AET(x_combo_initial,y_combo_initial,z_combo_initial)

checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(initial_L_1) + " " + str(initial_L_1_p) + " " + str(initial_L_2) + " " + str(initial_L_2_p)+ " " + str(initial_L_3) + " " + str(initial_L_3_p) + " " + str(accept/k) + "\n")
old_L_1 = initial_L_1
old_L_1_p = initial_L_1_p
old_L_2 = initial_L_2
old_L_2_p = initial_L_2_p
old_L_3 = initial_L_3
old_L_3_p = initial_L_3_p
x_combo_old = x_combo_initial
y_combo_old = y_combo_initial
z_combo_old = z_combo_initial

draw_count = 0
#rest of mcmc chain

while k <= number_chain:



	print('chain number')
	print(k)

	


	#sampling individual parameters

	#...........................L1........................................
	#...........................L1........................................
	#...........................L1........................................
	#L2 holding others fixed
	L_1_draw = proposal(old_L_1,draw_count)

	y_combo_new = y_combo(old_L_3,L_1_draw,old_L_3_p,old_L_1_p)
	z_combo_new = z_combo(old_L_2,L_1_draw,old_L_2_p,old_L_1_p)
	new_likelihood,new_determ_here,new_chi_2_here = likelihood_AET(x_combo_old,y_combo_new,z_combo_new)


	#alpha = min(new_likelihood-old_likelihood,0)
	alpha = min(np.log(prior(L_1_draw))+new_likelihood-np.log(prior(old_L_1))-old_likelihood,0)
	#alpha = min(np.log(prior(L_2_draw))+new_likelihood+np.log(q_top)-np.log(prior(old_L_2))-old_likelihood-np.log(q_bottom),0)

	u = np.log(np.random.uniform(0.000,1.000))

	if alpha >= u:
		old_L_1 = L_1_draw  #L_1_chain = np.append(L_1_chain,L_1_draw)
		old_likelihood = new_likelihood
		determ_here = new_determ_here
		chi_2_here = new_chi_2_here
		y_combo_old=y_combo_new
		z_combo_old=z_combo_new

		accept+=1
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")

	elif (alpha < u) or math.isnan(alpha):
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")

	else:
		print('something wrong in acceptance/rejection step')
	#next iteration
	k+=1	

	#...........................L1'........................................
	#...........................L1'........................................
	#...........................L1'........................................
	#L2 holding others fixed
	L_1_p_draw = proposal(old_L_1_p,draw_count)

	y_combo_new = y_combo(old_L_3,old_L_1,old_L_3_p,L_1_p_draw)
	z_combo_new = z_combo(old_L_2,old_L_1,old_L_2_p,L_1_p_draw)
	new_likelihood,new_determ_here,new_chi_2_here = likelihood_AET(x_combo_old,y_combo_new,z_combo_new)


	#alpha = min(new_likelihood-old_likelihood,0)
	alpha = min(np.log(prior(L_1_p_draw))+new_likelihood-np.log(prior(old_L_1_p))-old_likelihood,0)
	#alpha = min(np.log(prior(L_2_draw))+new_likelihood+np.log(q_top)-np.log(prior(old_L_2))-old_likelihood-np.log(q_bottom),0)

	u = np.log(np.random.uniform(0.000,1.000))

	if alpha >= u:
		old_L_1_p = L_1_p_draw  #L_1_chain = np.append(L_1_chain,L_1_draw)
		old_likelihood = new_likelihood
		determ_here = new_determ_here
		chi_2_here = new_chi_2_here
		y_combo_old=y_combo_new
		z_combo_old=z_combo_new

		accept+=1
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")


	elif (alpha < u) or math.isnan(alpha):
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")

	else:
		print('something wrong in acceptance/rejection step')
	#next iteration
	k+=1

	#...........................L2........................................
	#...........................L2........................................
	#...........................L2........................................
	#L2 holding others fixed
	L_2_draw = proposal(old_L_2,draw_count)

	x_combo_new = x_combo(old_L_3,L_2_draw,old_L_3_p,old_L_2_p)
	z_combo_new = z_combo(L_2_draw,old_L_1,old_L_2_p,old_L_1_p)
	new_likelihood,new_determ_here,new_chi_2_here = likelihood_AET(x_combo_new,y_combo_old,z_combo_new)


	#alpha = min(new_likelihood-old_likelihood,0)
	alpha = min(np.log(prior(L_2_draw))+new_likelihood-np.log(prior(old_L_2))-old_likelihood,0)
	#alpha = min(np.log(prior(L_2_draw))+new_likelihood+np.log(q_top)-np.log(prior(old_L_2))-old_likelihood-np.log(q_bottom),0)

	u = np.log(np.random.uniform(0.000,1.000))

	if alpha >= u:
		old_L_2 = L_2_draw  #L_1_chain = np.append(L_1_chain,L_1_draw)
		old_likelihood = new_likelihood
		determ_here = new_determ_here
		chi_2_here = new_chi_2_here
		x_combo_old=x_combo_new
		z_combo_old=z_combo_new

		accept+=1
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")

	elif (alpha < u) or math.isnan(alpha):
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")

	else:
		print('something wrong in acceptance/rejection step')
	#next iteration
	k+=1		

	#...........................L2'........................................
	#...........................L2'........................................
	#...........................L2'........................................
	#L2 holding others fixed
	L_2_p_draw = proposal(old_L_2_p,draw_count)

	x_combo_new = x_combo(old_L_3,old_L_2,old_L_3_p,L_2_p_draw)
	z_combo_new = z_combo(old_L_2,old_L_1,L_2_p_draw,old_L_1_p)
	new_likelihood,new_determ_here,new_chi_2_here = likelihood_AET(x_combo_new,y_combo_old,z_combo_new)


	#alpha = min(new_likelihood-old_likelihood,0)
	alpha = min(np.log(prior(L_2_p_draw))+new_likelihood-np.log(prior(old_L_2_p))-old_likelihood,0)
	#alpha = min(np.log(prior(L_2_draw))+new_likelihood+np.log(q_top)-np.log(prior(old_L_2))-old_likelihood-np.log(q_bottom),0)

	u = np.log(np.random.uniform(0.000,1.000))

	if alpha >= u:
		old_L_2_p = L_2_p_draw  #L_1_chain = np.append(L_1_chain,L_1_draw)
		old_likelihood = new_likelihood
		determ_here = new_determ_here
		chi_2_here = new_chi_2_here
		x_combo_old=x_combo_new
		z_combo_old=z_combo_new

		accept+=1
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")

	elif (alpha < u) or math.isnan(alpha):
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")

	else:
		print('something wrong in acceptance/rejection step')
	#next iteration
	k+=1	

	#...........................L3........................................
	#...........................L3........................................
	#...........................L3........................................


	#L3 holding others fixed
	L_3_draw = proposal(old_L_3,draw_count)


	x_combo_new = x_combo(L_3_draw,old_L_2,old_L_3_p,old_L_2_p)
	y_combo_new = y_combo(L_3_draw,old_L_1,old_L_3_p,old_L_1_p)
	#z_combo_new = z_combo_old

	new_likelihood,new_determ_here,new_chi_2_here = likelihood_AET(x_combo_new,y_combo_new,z_combo_old)



	#alpha = min(new_likelihood-old_likelihood,0)
	alpha = min(np.log(prior(L_3_draw))+new_likelihood-np.log(prior(old_L_3))-old_likelihood,0)
	#alpha = min(np.log(prior(L_3_draw))+new_likelihood+np.log(q_top)-np.log(prior(old_L_3))-old_likelihood-np.log(q_bottom),0)


	u = np.log(np.random.uniform(0.000,1.000))

	if alpha >= u:
		old_L_3 = L_3_draw  #L_1_chain = np.append(L_1_chain,L_1_draw)
		old_likelihood = new_likelihood
		determ_here = new_determ_here
		chi_2_here = new_chi_2_here
		x_combo_old = x_combo_new
		y_combo_old = y_combo_new

		accept+=1
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")

	elif (alpha < u) or math.isnan(alpha):
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")

	else:
		print('something wrong in acceptance/rejection step')
	#next iteration
	k+=1		

	#...........................L3'........................................
	#...........................L3'........................................
	#...........................L3'........................................

	#L3 holding others fixed
	L_3_p_draw = proposal(old_L_3_p,draw_count)

	x_combo_new = x_combo(old_L_3,old_L_2,L_3_p_draw,old_L_2_p)
	y_combo_new = y_combo(old_L_3,old_L_1,L_3_p_draw,old_L_1_p)
	#z_combo_new = z_combo_old

	new_likelihood,new_determ_here,new_chi_2_here = likelihood_AET(x_combo_new,y_combo_new,z_combo_old)



	#alpha = min(new_likelihood-old_likelihood,0)
	alpha = min(np.log(prior(L_3_p_draw))+new_likelihood-np.log(prior(old_L_3_p))-old_likelihood,0)
	#alpha = min(np.log(prior(L_3_draw))+new_likelihood+np.log(q_top)-np.log(prior(old_L_3))-old_likelihood-np.log(q_bottom),0)


	u = np.log(np.random.uniform(0.000,1.000))

	if alpha >= u:
		old_L_3_p = L_3_p_draw  #L_1_chain = np.append(L_1_chain,L_1_draw)
		old_likelihood = new_likelihood
		determ_here = new_determ_here
		chi_2_here = new_chi_2_here
		x_combo_old = x_combo_new
		y_combo_old = y_combo_new

		accept+=1
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")

	elif (alpha < u) or math.isnan(alpha):
		checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p) + " " + str(accept/k) + "\n")

	else:
		print('something wrong in acceptance/rejection step')
	#next iteration
	k+=1	

	


	if draw_count == 0 or draw_count ==1 or draw_count==2:
		draw_count+=1
		continue
	
	elif draw_count == 3:
		draw_count = 0	

	'''
	#Now only drawing between 2 proposal distributions; excluding uniform draws because want to avoid finite range
	if draw_count == 0:
		draw_count = 1
		continue
	elif draw_count == 1:
		draw_count = 0
	'''
checkfile.close()

print('acceptance ratio')
print(accept/number_chain)

print('number of new proposals accepted')
print(accept)

print("--- %s seconds ---" % (time.time() - start_time))

