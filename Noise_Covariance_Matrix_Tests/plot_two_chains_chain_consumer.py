import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import corner 
import matplotlib.lines as mlines
from chainconsumer import ChainConsumer


L_1_real = 8.339095192
L_1_p_real = 8.338994879
L_2_real = 8.338867041
L_2_p_real = 8.339095192
L_3_real = 8.338994879
L_3_p_real = 8.338867041

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
'''

likelihood_AET = data_AET_N_29['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_AET))
likelihood_AET = data_AET_N_29['likelihood'][removed::]
L_3_AET = data_AET_N_29['L_3'][removed::5]
L_2_AET = data_AET_N_29['L_2'][removed::5]
L_1_AET = data_AET_N_29['L_1'][removed::5]
L_3_p_AET = data_AET_N_29['L_3_p'][removed::5]
L_2_p_AET = data_AET_N_29['L_2_p'][removed::5]
L_1_p_AET = data_AET_N_29['L_1_p'][removed::5]
chi_2_AET = data_AET_N_29['sum_chi_2'][removed::5]

likelihood_equal = data_equal_29['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_equal))
likelihood_equal = data_equal_29['likelihood'][removed::5]
L_3_equal = data_equal_29['L_3'][removed::5]
L_2_equal = data_equal_29['L_2'][removed::5]
L_1_equal = data_equal_29['L_1'][removed::5]
L_3_p_equal = data_equal_29['L_3_p'][removed::5]
L_2_p_equal = data_equal_29['L_2_p'][removed::5]
L_1_p_equal = data_equal_29['L_1_p'][removed::5]
chi_2_equal = data_equal_29['sum_chi_2'][removed::5]

likelihood_unequal = data_unequal_N_49['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_unequal))
likelihood_unequal = data_unequal_N_49['likelihood'][removed::5]
L_3_unequal = data_unequal_N_49['L_3'][removed::5]
L_2_unequal = data_unequal_N_49['L_2'][removed::5]
L_1_unequal = data_unequal_N_49['L_1'][removed::5]
L_3_p_unequal = data_unequal_N_49['L_3_p'][removed::5]
L_2_p_unequal = data_unequal_N_49['L_2_p'][removed::5]
L_1_p_unequal = data_unequal_N_49['L_1_p'][removed::5]
chi_2_unequal = data_unequal_N_49['sum_chi_2'][removed::5]
'''
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



data_plot_unequal = np.array([L_3_here_unequal,L_2_here_unequal,L_1_here_unequal,L_3_p_here_unequal,L_2_p_here_unequal,L_1_p_here_unequal]).T

data_plot_equal = np.array([L_3_here_equal,L_2_here_equal,L_1_here_equal,L_3_p_here_equal,L_2_p_here_equal,L_1_p_here_equal]).T

data_plot_AET = np.array([L_3_here_AET,L_2_here_AET,L_1_here_AET,L_3_p_here_AET,L_2_p_here_AET,L_1_p_here_AET]).T





#----------------------------------------------------------------------------
#chain consumer method
#----------------------------------------------------------------------------

c = ChainConsumer()
parameters=[ r"$L_{3} \: (ns)$", r"$L_{2} \: (ns)$",r"$L_{1} \: (ns)$", r"$L^{'}_{3} \: (ns)$",r"$L^{'}_{2} \: (ns)$",r"$L^{'}_{1} \: (ns)$"]
#parameters=[ r"$L_{3}-L_{3_{True}}$", r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"]
c.add_chain(data_plot_AET, parameters=parameters,color='purple', name = 'AET')
c.add_chain(data_plot_equal, parameters=parameters, color='blue', name = 'Equal-arm XYZ')
c.add_chain(data_plot_unequal, parameters=parameters, color='green', name = 'Unequal-arm XYZ')

'''
c.add_chain(data_plot_43, parameters=parameters,color='blue', name='LaGrange N=43')
c.add_chain(data_plot_41, parameters=parameters,color='green', name ='LaGrange N=41')
c.add_chain(data_plot_37, parameters=parameters,color='purple', name = 'LaGrange N=37')
c.add_chain(data_plot_35, parameters=parameters,color='orange', name = 'LaGrange N=35')
c.add_chain(data_plot_33, parameters=parameters,color='red', name = 'LaGrange N=33')
c.add_chain(data_plot_31, parameters=parameters, name = 'LaGrange N=31')
c.add_chain(data_plot_29, parameters=parameters, name = 'LaGrange N=29')
c.add_chain(data_plot_27, parameters=parameters, name = 'LaGrange N=27')
c.add_chain(data_plot_25, parameters=parameters, name = 'LaGrange N=25')
#c.add_chain(data_plot_23, parameters=parameters, name = 'LaGrange N=23')
c.add_chain(data_plot_21, parameters=parameters, name = 'LaGrange N=21')
'''
#90% credible interval, kde=False means no parameter smoothing
c.configure(sigmas=[0,1.645],kde=False,smooth=1,summary=True)
#c.configure(sigmas=[0,1.645])
#c.plotter.plot_summary(chains=['LaGrange N=59','LaGrange N=49','LaGrange N=43','LaGrange N=41','LaGrange N=37','LaGrange N=35','LaGrange N=33'],filename='summary_plot_chain_consumer_N=43_vs_41_vs_37_vs_35_vs_33_vs_49_vs_59.png',display=True, truth=[0e1,0e1,0e1,0e1,0e1,0e1])
#c.plotter.plot(chains=['LaGrange N=59','LaGrange N=49','LaGrange N=43','LaGrange N=41','LaGrange N=37','LaGrange N=35','LaGrange N=33','LaGrange N=31','LaGrange N=29','LaGrange N=27','LaGrange N=25','LaGrange N=21'],filename='comparison_chain_consumer_N=43_vs_41_vs_37_vs_35_vs_33_vs_49_vs_59_vs 31_vs_29_vs_27_vs_25_vs_21.png',display=True, truth=[0e1,0e1,0e1,0e1,0e1,0e1], legend=True)
c.plotter.plot(chains=['AET','Equal-arm XYZ','Unequal-arm XYZ'],filename='chain_consumer_equal_vs_unequal_vs_AET.png',display=True, truth=[0e1,0e1,0e1,0e1,0e1,0e1], legend=True)

c.analysis.get_latex_table(parameters=parameters, hlines=True, blank_fill='--', filename='noise_cov_matrix_table')



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