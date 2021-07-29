import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color 
from mpl_toolkits.mplot3d import Axes3D
import corner 
import matplotlib.lines as mlines
from chainconsumer import ChainConsumer
import statsmodels.api as sm

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

L_1_real = 8.339095192
L_1_p_real = 8.338994879
L_2_real = 8.338867041
L_2_p_real = 8.339095192
L_3_real = 8.338994879
L_3_p_real = 8.338867041


data_59 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_59.dat',names = True)

data_49 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_49.dat',names = True)

data_43 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_43.dat',names = True)

data_41 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_41.dat',names = True)

data_39 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_39.dat',names = True)

data_37 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_37.dat',names = True)


data_35 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_35.dat',names = True)

data_33 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_33.dat',names = True)

data_31 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_31.dat',names = True)

data_29 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_29.dat',names = True)

data_27 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_27.dat',names = True)

data_25 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_25.dat',names = True)

data_23 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_23.dat',names = True)

data_21 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_21.dat',names = True)

data_19 = np.recfromtxt('./MCMC/chainfile_noise_matrix_equal_arm_967_19.dat',names = True)




dir = '.'

likelihood_59 = data_59['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_59))
likelihood_59 = data_59['likelihood'][removed::]
L_3_59 = data_59['L_3'][removed::]
L_2_59 = data_59['L_2'][removed::]
L_1_59 = data_59['L_1'][removed::]
L_3_p_59 = data_59['L_3_p'][removed::]
L_2_p_59 = data_59['L_2_p'][removed::]
L_1_p_59 = data_59['L_1_p'][removed::]
chi_2_59 = data_59['sum_chi_2'][removed::]

likelihood_49 = data_49['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_49))
likelihood_49 = data_49['likelihood'][removed::]
L_3_49 = data_49['L_3'][removed::]
L_2_49 = data_49['L_2'][removed::]
L_1_49 = data_49['L_1'][removed::]
L_3_p_49 = data_49['L_3_p'][removed::]
L_2_p_49 = data_49['L_2_p'][removed::]
L_1_p_49 = data_49['L_1_p'][removed::]
chi_2_49 = data_49['sum_chi_2'][removed::]

likelihood_43 = data_43['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_43))
likelihood_43 = data_43['likelihood'][removed::]
L_3_43 = data_43['L_3'][removed::]
L_2_43 = data_43['L_2'][removed::]
L_1_43 = data_43['L_1'][removed::]
L_3_p_43 = data_43['L_3_p'][removed::]
L_2_p_43 = data_43['L_2_p'][removed::]
L_1_p_43 = data_43['L_1_p'][removed::]
chi_2_43 = data_43['sum_chi_2'][removed::]

likelihood_41 = data_41['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_41))
likelihood_41 = data_41['likelihood'][removed::]
L_3_41 = data_41['L_3'][removed::]
L_2_41 = data_41['L_2'][removed::]
L_1_41 = data_41['L_1'][removed::]
L_3_p_41 = data_41['L_3_p'][removed::]
L_2_p_41 = data_41['L_2_p'][removed::]
L_1_p_41 = data_41['L_1_p'][removed::]
chi_2_41 = data_41['sum_chi_2'][removed::]

likelihood_39 = data_39['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_39))
likelihood_39 = data_39['likelihood'][removed::]
L_3_39 = data_39['L_3'][removed::]
L_2_39 = data_39['L_2'][removed::]
L_1_39 = data_39['L_1'][removed::]
L_3_p_39 = data_39['L_3_p'][removed::]
L_2_p_39 = data_39['L_2_p'][removed::]
L_1_p_39 = data_39['L_1_p'][removed::]
chi_2_39 = data_39['sum_chi_2'][removed::]




likelihood_37 = data_37['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_37))
likelihood_37 = data_37['likelihood'][removed::]
L_3_37 = data_37['L_3'][removed::]
L_2_37 = data_37['L_2'][removed::]
L_1_37 = data_37['L_1'][removed::]
L_3_p_37 = data_37['L_3_p'][removed::]
L_2_p_37 = data_37['L_2_p'][removed::]
L_1_p_37 = data_37['L_1_p'][removed::]
chi_2_37 = data_37['sum_chi_2'][removed::]

likelihood_35 = data_35['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_35))
likelihood_35 = data_35['likelihood'][removed::]
L_3_35 = data_35['L_3'][removed::]
L_2_35 = data_35['L_2'][removed::]
L_1_35 = data_35['L_1'][removed::]
L_3_p_35 = data_35['L_3_p'][removed::]
L_2_p_35 = data_35['L_2_p'][removed::]
L_1_p_35 = data_35['L_1_p'][removed::]
chi_2_35 = data_35['sum_chi_2'][removed::]

likelihood_33 = data_33['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_33))
likelihood_33 = data_33['likelihood'][removed::]
L_3_33 = data_33['L_3'][removed::]
L_2_33 = data_33['L_2'][removed::]
L_1_33 = data_33['L_1'][removed::]
L_3_p_33 = data_33['L_3_p'][removed::]
L_2_p_33 = data_33['L_2_p'][removed::]
L_1_p_33 = data_33['L_1_p'][removed::]
chi_2_33 = data_33['sum_chi_2'][removed::]

likelihood_31 = data_31['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_31))
likelihood_31 = data_31['likelihood'][removed::]
L_3_31 = data_31['L_3'][removed::]
L_2_31 = data_31['L_2'][removed::]
L_1_31 = data_31['L_1'][removed::]
L_3_p_31 = data_31['L_3_p'][removed::]
L_2_p_31 = data_31['L_2_p'][removed::]
L_1_p_31 = data_31['L_1_p'][removed::]
chi_2_31 = data_31['sum_chi_2'][removed::]

likelihood_29 = data_29['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_29))
likelihood_29 = data_29['likelihood'][removed::]
L_3_29 = data_29['L_3'][removed::]
L_2_29 = data_29['L_2'][removed::]
L_1_29 = data_29['L_1'][removed::]
L_3_p_29 = data_29['L_3_p'][removed::]
L_2_p_29 = data_29['L_2_p'][removed::]
L_1_p_29 = data_29['L_1_p'][removed::]
chi_2_29 = data_29['sum_chi_2'][removed::]


likelihood_27 = data_27['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_27))
likelihood_27 = data_27['likelihood'][removed::]
L_3_27 = data_27['L_3'][removed::]
L_2_27 = data_27['L_2'][removed::]
L_1_27 = data_27['L_1'][removed::]
L_3_p_27 = data_27['L_3_p'][removed::]
L_2_p_27 = data_27['L_2_p'][removed::]
L_1_p_27 = data_27['L_1_p'][removed::]
chi_2_27 = data_27['sum_chi_2'][removed::]

likelihood_25 = data_25['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_25))
likelihood_25 = data_25['likelihood'][removed::]
L_3_25 = data_25['L_3'][removed::]
L_2_25 = data_25['L_2'][removed::]
L_1_25 = data_25['L_1'][removed::]
L_3_p_25 = data_25['L_3_p'][removed::]
L_2_p_25 = data_25['L_2_p'][removed::]
L_1_p_25 = data_25['L_1_p'][removed::]
chi_2_25 = data_25['sum_chi_2'][removed::]

likelihood_23 = data_23['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_23))
likelihood_23 = data_23['likelihood'][removed::]
L_3_23 = data_23['L_3'][removed::]
L_2_23 = data_23['L_2'][removed::]
L_1_23 = data_23['L_1'][removed::]
L_3_p_23 = data_23['L_3_p'][removed::]
L_2_p_23 = data_23['L_2_p'][removed::]
L_1_p_23 = data_23['L_1_p'][removed::]
chi_2_23 = data_23['sum_chi_2'][removed::]

likelihood_21 = data_21['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_21))
likelihood_21 = data_21['likelihood'][removed::]
L_3_21 = data_21['L_3'][removed::]
L_2_21 = data_21['L_2'][removed::]
L_1_21 = data_21['L_1'][removed::]
L_3_p_21 = data_21['L_3_p'][removed::]
L_2_p_21 = data_21['L_2_p'][removed::]
L_1_p_21 = data_21['L_1_p'][removed::]
chi_2_21 = data_21['sum_chi_2'][removed::]

likelihood_19 = data_19['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_19))
likelihood_19 = data_19['likelihood'][removed::]
L_3_19 = data_19['L_3'][removed::]
L_2_19 = data_19['L_2'][removed::]
L_1_19 = data_19['L_1'][removed::]
L_3_p_19 = data_19['L_3_p'][removed::]
L_2_p_19 = data_19['L_2_p'][removed::]
L_1_p_19 = data_19['L_1_p'][removed::]
chi_2_19 = data_19['sum_chi_2'][removed::]


L_3_here_59 = (L_3_59-L_3_real)*1e9
L_2_here_59 = (L_2_59-L_2_real)*1e9
L_1_here_59 = (L_1_59-L_1_real)*1e9
L_3_p_here_59 = (L_3_p_59-L_3_p_real)*1e9
L_2_p_here_59 = (L_2_p_59-L_2_p_real)*1e9
L_1_p_here_59 = (L_1_p_59-L_1_p_real)*1e9

L_3_here_49 = (L_3_49-L_3_real)*1e9
L_2_here_49 = (L_2_49-L_2_real)*1e9
L_1_here_49 = (L_1_49-L_1_real)*1e9
L_3_p_here_49 = (L_3_p_49-L_3_p_real)*1e9
L_2_p_here_49 = (L_2_p_49-L_2_p_real)*1e9
L_1_p_here_49 = (L_1_p_49-L_1_p_real)*1e9


L_3_here_43 = (L_3_43-L_3_real)*1e9
L_2_here_43 = (L_2_43-L_2_real)*1e9
L_1_here_43 = (L_1_43-L_1_real)*1e9
L_3_p_here_43 = (L_3_p_43-L_3_p_real)*1e9
L_2_p_here_43 = (L_2_p_43-L_2_p_real)*1e9
L_1_p_here_43 = (L_1_p_43-L_1_p_real)*1e9

L_3_here_41 = (L_3_41-L_3_real)*1e9
L_2_here_41 = (L_2_41-L_2_real)*1e9
L_1_here_41 = (L_1_41-L_1_real)*1e9
L_3_p_here_41 = (L_3_p_41-L_3_p_real)*1e9
L_2_p_here_41 = (L_2_p_41-L_2_p_real)*1e9
L_1_p_here_41 = (L_1_p_41-L_1_p_real)*1e9

L_3_here_39 = (L_3_39-L_3_real)*1e9
L_2_here_39 = (L_2_39-L_2_real)*1e9
L_1_here_39 = (L_1_39-L_1_real)*1e9
L_3_p_here_39 = (L_3_p_39-L_3_p_real)*1e9
L_2_p_here_39 = (L_2_p_39-L_2_p_real)*1e9
L_1_p_here_39 = (L_1_p_39-L_1_p_real)*1e9

L_3_here_37 = (L_3_37-L_3_real)*1e9
L_2_here_37 = (L_2_37-L_2_real)*1e9
L_1_here_37 = (L_1_37-L_1_real)*1e9
L_3_p_here_37 = (L_3_p_37-L_3_p_real)*1e9
L_2_p_here_37 = (L_2_p_37-L_2_p_real)*1e9
L_1_p_here_37 = (L_1_p_37-L_1_p_real)*1e9

L_3_here_35 = (L_3_35-L_3_real)*1e9
L_2_here_35 = (L_2_35-L_2_real)*1e9
L_1_here_35 = (L_1_35-L_1_real)*1e9
L_3_p_here_35 = (L_3_p_35-L_3_p_real)*1e9
L_2_p_here_35 = (L_2_p_35-L_2_p_real)*1e9
L_1_p_here_35 = (L_1_p_35-L_1_p_real)*1e9

L_3_here_33 = (L_3_33-L_3_real)*1e9
L_2_here_33 = (L_2_33-L_2_real)*1e9
L_1_here_33 = (L_1_33-L_1_real)*1e9
L_3_p_here_33 = (L_3_p_33-L_3_p_real)*1e9
L_2_p_here_33 = (L_2_p_33-L_2_p_real)*1e9
L_1_p_here_33 = (L_1_p_33-L_1_p_real)*1e9

L_3_here_31 = (L_3_31-L_3_real)*1e9
L_2_here_31 = (L_2_31-L_2_real)*1e9
L_1_here_31 = (L_1_31-L_1_real)*1e9
L_3_p_here_31 = (L_3_p_31-L_3_p_real)*1e9
L_2_p_here_31 = (L_2_p_31-L_2_p_real)*1e9
L_1_p_here_31 = (L_1_p_31-L_1_p_real)*1e9

L_3_here_29 = (L_3_29-L_3_real)*1e9
L_2_here_29 = (L_2_29-L_2_real)*1e9
L_1_here_29 = (L_1_29-L_1_real)*1e9
L_3_p_here_29 = (L_3_p_29-L_3_p_real)*1e9
L_2_p_here_29 = (L_2_p_29-L_2_p_real)*1e9
L_1_p_here_29 = (L_1_p_29-L_1_p_real)*1e9

L_3_here_27 = (L_3_27-L_3_real)*1e9
L_2_here_27 = (L_2_27-L_2_real)*1e9
L_1_here_27 = (L_1_27-L_1_real)*1e9
L_3_p_here_27 = (L_3_p_27-L_3_p_real)*1e9
L_2_p_here_27 = (L_2_p_27-L_2_p_real)*1e9
L_1_p_here_27 = (L_1_p_27-L_1_p_real)*1e9

L_3_here_25 = (L_3_25-L_3_real)*1e9
L_2_here_25 = (L_2_25-L_2_real)*1e9
L_1_here_25 = (L_1_25-L_1_real)*1e9
L_3_p_here_25 = (L_3_p_25-L_3_p_real)*1e9
L_2_p_here_25 = (L_2_p_25-L_2_p_real)*1e9
L_1_p_here_25 = (L_1_p_25-L_1_p_real)*1e9

L_3_here_23 = (L_3_23-L_3_real)*1e9
L_2_here_23 = (L_2_23-L_2_real)*1e9
L_1_here_23 = (L_1_23-L_1_real)*1e9
L_3_p_here_23 = (L_3_p_23-L_3_p_real)*1e9
L_2_p_here_23 = (L_2_p_23-L_2_p_real)*1e9
L_1_p_here_23 = (L_1_p_23-L_1_p_real)*1e9

L_3_here_21 = (L_3_21-L_3_real)*1e9
L_2_here_21 = (L_2_21-L_2_real)*1e9
L_1_here_21 = (L_1_21-L_1_real)*1e9
L_3_p_here_21 = (L_3_p_21-L_3_p_real)*1e9
L_2_p_here_21 = (L_2_p_21-L_2_p_real)*1e9
L_1_p_here_21 = (L_1_p_21-L_1_p_real)*1e9

L_3_here_19 = (L_3_19-L_3_real)*1e9
L_2_here_19 = (L_2_19-L_2_real)*1e9
L_1_here_19 = (L_1_19-L_1_real)*1e9
L_3_p_here_19 = (L_3_p_19-L_3_p_real)*1e9
L_2_p_here_19 = (L_2_p_19-L_2_p_real)*1e9
L_1_p_here_19 = (L_1_p_19-L_1_p_real)*1e9

ndim, nsamples = 5, len(likelihood_49)

data_plot_59 = np.array([L_3_here_59,L_2_here_59,L_1_here_59,L_3_p_here_59,L_2_p_here_59,L_1_p_here_59]).T

data_plot_49 = np.array([L_3_here_49,L_2_here_49,L_1_here_49,L_3_p_here_49,L_2_p_here_49,L_1_p_here_49]).T

data_plot_43 = np.array([L_3_here_43,L_2_here_43,L_1_here_43,L_3_p_here_43,L_2_p_here_43,L_1_p_here_43]).T

data_plot_41 = np.array([L_3_here_41,L_2_here_41,L_1_here_41,L_3_p_here_41,L_2_p_here_41,L_1_p_here_41]).T

data_plot_39 = np.array([L_3_here_39,L_2_here_39,L_1_here_39,L_3_p_here_39,L_2_p_here_39,L_1_p_here_39]).T

data_plot_37 = np.array([L_3_here_37,L_2_here_37,L_1_here_37,L_3_p_here_37,L_2_p_here_37,L_1_p_here_37]).T

data_plot_35 = np.array([L_3_here_35,L_2_here_35,L_1_here_35,L_3_p_here_35,L_2_p_here_35,L_1_p_here_35]).T

data_plot_33 = np.array([L_3_here_33,L_2_here_33,L_1_here_33,L_3_p_here_33,L_2_p_here_33,L_1_p_here_33]).T

data_plot_31 = np.array([L_3_here_31,L_2_here_31,L_1_here_31,L_3_p_here_31,L_2_p_here_31,L_1_p_here_31]).T

data_plot_29 = np.array([L_3_here_29,L_2_here_29,L_1_here_29,L_3_p_here_29,L_2_p_here_29,L_1_p_here_29]).T

data_plot_27 = np.array([L_3_here_27,L_2_here_27,L_1_here_27,L_3_p_here_27,L_2_p_here_27,L_1_p_here_27]).T

data_plot_25 = np.array([L_3_here_25,L_2_here_25,L_1_here_25,L_3_p_here_25,L_2_p_here_25,L_1_p_here_25]).T

data_plot_23 = np.array([L_3_here_23,L_2_here_23,L_1_here_23,L_3_p_here_23,L_2_p_here_23,L_1_p_here_23]).T

data_plot_21 = np.array([L_3_here_21,L_2_here_21,L_1_here_21,L_3_p_here_21,L_2_p_here_21,L_1_p_here_21]).T

data_plot_19 = np.array([L_3_here_19,L_2_here_19,L_1_here_19,L_3_p_here_19,L_2_p_here_19,L_1_p_here_19]).T

#chi_2_data_plot = np.array([chi_2_59,chi_2_49,chi_2_41,chi_2_43,chi_2_37,chi_2_35,chi_2_33,chi_2_31,chi_2_29,chi_2_27,chi_2_25,chi_2_21]).T


#----------------------------------------------------------------------------
#chain consumer method
#----------------------------------------------------------------------------

c = ChainConsumer()
parameters=[ r"$\delta L_{3} \: (ns)$", r"$\delta L_{2} \: (ns)$",r"$\delta L_{1} \: (ns)$", r"$\delta L^{'}_{3} \: (ns)$",r"$\delta L^{'}_{2} \: (ns)$",r"$\delta L^{'}_{1} \: (ns)$"]

#c.add_chain(data_plot_59, parameters=parameters,color='cyan', name = 'LaGrange N=59')
c.add_chain(data_plot_49, parameters=parameters, color=color.to_hex('teal'), name = 'N=49')
#c.add_chain(data_plot_43, parameters=parameters,color='blue', name='LaGrange N=43')
#c.add_chain(data_plot_41, parameters=parameters,color='green', name ='LaGrange N=41')
c.add_chain(data_plot_39, parameters=parameters,color=color.to_hex('darkviolet'), name = 'N=39')
#c.add_chain(data_plot_37, parameters=parameters,color='purple', name = 'LaGrange N=37')
#c.add_chain(data_plot_35, parameters=parameters,color=color.to_hex('darkorange'), name = 'LaGrange N=35')
#c.add_chain(data_plot_33, parameters=parameters,color='black', name = 'LaGrange N=33')
c.add_chain(data_plot_29, parameters=parameters,color='orange', name = 'N=29')
c.add_chain(data_plot_27, parameters=parameters,color=color.to_hex('deeppink'), name = 'N=27')
c.add_chain(data_plot_25, parameters=parameters,color=color.to_hex('lime'), name = 'N=25')
c.add_chain(data_plot_23, parameters=parameters, color=color.to_hex('royalblue'),name = 'N=23')
c.add_chain(data_plot_21, parameters=parameters,color=color.to_hex('firebrick'), name = 'N=21')
c.add_chain(data_plot_19, parameters=parameters,color=color.to_hex('darkgreen'), name = 'N=19')

#colormap = ['black','magenta','black','black','indigo','black','black','black','lime','black','deepskyblue','brown','deeppink','dodgerblue']
#90% credible interval, kde=False means no parameter smoothing
#c.configure(sigmas=[0,1.645],kde=False,smooth=1,summary=True)
#c.configure(sigmas=[0,1.645])
c.configure(sigmas=[0,1.645],spacing=1.0,kde=False,smooth=1,usetex=True,shade=False,legend_kwargs={'fontsize':'x-large','bbox_to_anchor':(-2, 0.9)})

#fig= plt.figure()
#c.plotter.plot_summary(chains=['LaGrange N=59','LaGrange N=49','LaGrange N=43','LaGrange N=41','LaGrange N=37','LaGrange N=35','LaGrange N=33'],filename='summary_plot_chain_consumer_N=43_vs_41_vs_37_vs_35_vs_33_vs_49_vs_59.png',display=True, truth=[0e1,0e1,0e1,0e1,0e1,0e1])
#c.plotter.plot(chains=['LaGrange N=59','LaGrange N=49','LaGrange N=43','LaGrange N=41','LaGrange N=39','LaGrange N=37','LaGrange N=35','LaGrange N=33','LaGrange N=31','LaGrange N=29','LaGrange N=27','LaGrange N=25','LaGrange N=23','LaGrange N=21','LaGrange N=19'],filename='comparison_chain_consumer_uniform_filter_cut.png', truth=[0e1,0e1,0e1,0e1,0e1,0e1],display=True,legend=True)
#fig = c.plotter.plot(chains=['LaGrange N=49','LaGrange N=39','LaGrange N=29','LaGrange N=27','LaGrange N=25','LaGrange N=23','LaGrange N=21','LaGrange N=19'],filename='comparison_chain_consumer_uniform_filter_cut_49_39_35_29_27_25_23_21_19.png', truth=[0e1,0e1,0e1,0e1,0e1,0e1],display=True,legend=True)
fig = c.plotter.plot(chains=['N=49','N=39','N=29','N=27','N=25','N=23','N=21','N=19'], truth=[0e1,0e1,0e1,0e1,0e1,0e1],display=False,filename = 'fig_1_amaldi.png',legend=True)
print('ax')
print(fig.axes)
print('length of axes list')
print(len(fig.axes))
print('output of fig.legend')
print(fig.legend)
plt.close()
#c.plotter.plot(chains=['LaGrange N=29','LaGrange N=27','LaGrange N=25','LaGrange N=23','LaGrange N=21','LaGrange N=19'],filename='comparison_chain_consumer_N=29_vs_27_vs_25_vs_23_vs_21_vs_19.png', truth=[0e1,0e1,0e1,0e1,0e1,0e1],display=True,legend=True)
#fig.ticklabel_format(axis='both', style='sci')
#plt.show()
#c.plotter.plot(chains=['LaGrange N=49'],filename='chain_consumer_N=49.png',display=True, truth=[0e1,0e1,0e1,0e1,0e1,0e1], legend=True)
#c.diagnostic.gelman_rubin(chain=['LaGrange N=59'])
#c.analysis.get_latex_table(filename='filter_length_tests_summaries_uniform_filter_cut_selection')



#----------------------------------------------------------------------------
#chi^2 histograms
#----------------------------------------------------------------------------

colormap = ['black','teal','black','black','darkviolet','black','black','black','black','orange','deeppink','lime','royalblue','firebrick','darkgreen']


x = [59,49,43,41,39,37,35,33,31,29,27,25,23,21,19]
y = [np.median(chi_2_59),np.median(chi_2_49),np.median(chi_2_43),np.median(chi_2_41),np.median(chi_2_39),np.median(chi_2_37),np.median(chi_2_35),np.median(chi_2_33),np.median(chi_2_31),np.median(chi_2_29),np.median(chi_2_27),np.median(chi_2_25),np.median(chi_2_23),np.median(chi_2_21),np.median(chi_2_19)]

plt.scatter(x,y,c=colormap,marker='x')
plt.plot(x,y,c='k',alpha=0.7,linestyle='--')
plt.xlabel('Filter Length N')
plt.ylabel(r'median $\chi^{2}$')
plt.savefig('fig_2_amaldi.png')
plt.show()
#plt.close()

