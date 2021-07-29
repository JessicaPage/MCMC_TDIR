import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import corner 
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import matplotlib.colors as color 
from chainconsumer import ChainConsumer
#from matplotlib import rcParams
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (7,7)

L_1_real = 8.339095192
L_1_p_real = 8.338994879
L_2_real = 8.338867041
L_2_p_real = 8.339095192
L_3_real = 8.338994879
L_3_p_real = 8.338867041

#data = np.recfromtxt('chain_data_overlap_add.dat',names = True)
#data = np.recfromtxt('/Users/jessica/Desktop/Project_1/Rigid_Rotation/MCMC/chainfile.dat',names = True)

data_new = np.recfromtxt('chainfile_noise_matrix_equal_arm_940.dat',names = True)

data_old = np.recfromtxt('chainfile_noise_matrix_equal_arm_941.dat',names = True)

#data_true = np.recfromtxt('/Users/jessica/Desktop/Project_1/Rigid_Rotation/MCMC/new_simulation/include_2nd_noises/chainfile_noise_matrix_unequal_arm_793.dat',names = True)

dir = '.'

#data = np.genfromtxt('/Users/jessica/Desktop/Project_1/Rigid_Rotation/MCMC/chainfile.dat',skip_header=1)

print('data new')
print(data_new)
likelihood_new = data_new['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_new))

likelihood_new = data_new['likelihood'][removed::]
L_3_new = data_new['L_3'][removed::]
L_2_new = data_new['L_2'][removed::]
L_1_new = data_new['L_1'][removed::]

L_3_p_new = data_new['L_3_p'][removed::]
L_2_p_new = data_new['L_2_p'][removed::]
L_1_p_new = data_new['L_1_p'][removed::]

chi_2_new = data_new['sum_chi_2'][removed::]
log_term_determ_new = data_new['log_term_determ'][removed::]

print('data new')
print(data_new)
likelihood_old = data_old['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood_old))

likelihood_old = data_old['likelihood'][removed::]
L_3_old = data_old['L_3'][removed::]
L_2_old = data_old['L_2'][removed::]
L_1_old = data_old['L_1'][removed::]

L_3_p_old = data_old['L_3_p'][removed::]
L_2_p_old = data_old['L_2_p'][removed::]
L_1_p_old = data_old['L_1_p'][removed::]

chi_2_old = data_old['sum_chi_2'][removed::]
log_term_determ_old = data_old['log_term_determ'][removed::]






length = len(likelihood_old)



L_3_here = (L_3_new-L_3_real)*1e9
L_2_here = (L_2_new-L_2_real)*1e9
L_1_here = (L_1_new-L_1_real)*1e9
L_3_p_here = (L_3_p_new-L_3_p_real)*1e9
L_2_p_here = (L_2_p_new-L_2_p_real)*1e9
L_1_p_here = (L_1_p_new-L_1_p_real)*1e9


L_3_here_31 = (L_3_old-L_3_real)*1e9
L_2_here_31 = (L_2_old-L_2_real)*1e9
L_1_here_31 = (L_1_old-L_1_real)*1e9
L_3_p_here_31 = (L_3_p_old-L_3_p_real)*1e9
L_2_p_here_31 = (L_2_p_old-L_2_p_real)*1e9
L_1_p_here_31 = (L_1_p_old-L_1_p_real)*1e9






ndim, nsamples = 5, len(likelihood_new)

#data_plot = np.array([likelihood,L_3_here,L_2_here,L_3_p_here,L_2_p_here]).T

data_plot = np.array([L_3_here,L_2_here,L_1_here,L_3_p_here,L_2_p_here,L_1_p_here]).T

data_plot_31 = np.array([L_3_here_31,L_2_here_31,L_1_here_31,L_3_p_here_31,L_2_p_here_31,L_1_p_here_31]).T
'''
data_plot = np.array([L_3_new,L_2_new,L_1_new,L_3_p_new,L_2_p_new,L_1_p_new]).T

data_plot_31 = np.array([L_3_old,L_2_old,L_1_old,L_3_p_old,L_2_p_old,L_1_p_old]).T
'''
#----------------------------------------------------------------------------
#chain consumer method
#----------------------------------------------------------------------------
'''
from matplotlib import rcParams
rcParams['text.latex.preamble'] = r'\newcommand{\mathdefault}[1][]{}'
'''
'''
ticks_here = [-4e-8,-2e-8,0e1,2e-8,4e-8]
#labels = ['%.0E' % x for x in ticks_here]
labels = ['-4', '-2', '0', '2', '4']

ticks_here = [-4e-8,-2e-8,0e1,2e-8,4e-8]
labels =[r'-$4\times10^{-8}$', r'-$2\times10^{-8}$', '0', r'$2\times10^{-8}$', r'$4\times10^{-8}$']
'''
ticks_here = [-4e-8,-2e-8,0e1,2e-8,4e-8]
labels = ['-4', '-2', '0', '2', '4']

ticks_last = [-4e-8,-2e-8,0e1,4e-8]
labels_last =['-4', '-2','0',r'$4\times10^{-8}$']

ticks_last_y = [-4e-8,-2e-8,4e-8]
labels_last_y =['-4', '-2',r'$4\times10^{-8}$']
#fig, ax = plt.subplots(nrows=1, ncols=1)
#plt.ticklabel_format(useOffset=False,axis='both', style='sci')

c = ChainConsumer()
parameters=[ r"$\delta L_{3} \: (ns)$", r"$\delta L_{2} \: (ns)$",r"$\delta L_{1} \: (ns)$", r"$\delta L^{'}_{3} \: (ns)$",r"$\delta L^{'}_{2} \: (ns)$",r"$\delta L^{'}_{1} \: (ns)$"]

#parameters=[ r"$L_{3} \: ns)$", r"$L_{2} \: (ns)$",r"$L_{1} \: (ns)$", r"$L^{'}_{3} \: (ns)$",r"$L^{'}_{2} \: (ns)$",r"$L^{'}_{1} \: (ns)$"]
c.add_chain(data_plot, parameters=parameters,color=color.to_hex('green'),name='Oversampled Data Generation')
c.add_chain(data_plot_31, parameters=parameters,color=color.to_hex('blue'),name='Interpolated-only data')
c.configure(sigmas=[0,1.645],spacing=1.0,kde=False,smooth=1,shade=False)
#c.configure( max_ticks=8,usetex=True,diagonal_tick_labels=True, tick_font_size=8, label_font_size=25)
#c.analysis.get_summary(parameters=parameters,chains=['Oversampled Data Generation','Interpolated-only data'])
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
#ax1.ticklabel_format(axis='both', style='sci')
fig =c.plotter.plot(chains=['Oversampled Data Generation','Interpolated-only data'],figsize=(7,7),filename='comparison_chain_consumer.png', truth=[0e1,0e1,0e1,0e1,0e1,0e1],display=False,legend=True)
print('ax')
print(fig.axes[6])

print('length of axes list')
print(len(fig.axes))
plt.close()
'''
plt.sca(fig.axes[35])
plt.xticks(ticks_here, labels,rotation=None)
plt.sca(fig.axes[34])
plt.xticks(ticks_here, labels,rotation=None)
plt.sca(fig.axes[33])
plt.xticks(ticks_here, labels,rotation=None)
plt.sca(fig.axes[32])
plt.xticks(ticks_here, labels,rotation=None)
plt.sca(fig.axes[31])
plt.xticks(ticks_here, labels,rotation=None)
plt.sca(fig.axes[30])
plt.xticks(ticks_here, labels,rotation=None)
plt.sca(fig.axes[6])
plt.yticks(ticks_here, labels,rotation=90)
plt.sca(fig.axes[12])
plt.yticks(ticks_here, labels,rotation=90)
plt.sca(fig.axes[18])
plt.yticks(ticks_here, labels,rotation=90)
plt.sca(fig.axes[24])
plt.yticks(ticks_here, labels,rotation=90)
plt.sca(fig.axes[30])
plt.yticks(ticks_here, labels,rotation=90)
'''
#ax[0].set_yticklabels(labels)
#ax[0].set_major_formatter(mtick.FormatStrFormatter('%.2e'))
#fig=c.plotter.plot(chains=['Oversampled Data Generation','Interpolated-only data'],filename='comparison_chain_consumer_shifted.png', truth=[L_3_real,L_2_real,L_1_real,L_3_p_real,L_2_p_real,L_1_p_real],display=False,legend=True)
#c.plotter.plot_distributions(truth=[L_3_real,L_2_real,L_1_real,L_3_p_real,L_2_p_real,L_1_p_real], display=True)
#ax.set_size_inches(3 + fig.get_size_inches())
#plt.ticklabel_format(useOffset=False,axis='both', style='sci')
#plt.setp(fig.axes,xticks=ticks_here, xticklabels=labels)
#plt.xticks(ticks_here, labels)
#plt.yticks(ticks_here, labels)
#plt.show()


#----------------------------------------------------------------------------
#chi^2 histograms
#----------------------------------------------------------------------------
sixteen_old = np.quantile(chi_2_old,0.16)
sixteen_new = np.quantile(chi_2_new,0.16)

eightyfour_old =  np.quantile(chi_2_old,0.84)
eightyfour_new =  np.quantile(chi_2_new,0.84)

'''
plt.hist(chi_2_new,label='over-sampled data generation')
plt.hist(chi_2_old,label='interpolation-only data generation')
plt.axvline(sixteen_old, color='k', linestyle='dashed')
plt.axvline(sixteen_new, color='k', linestyle='dashed')
plt.axvline(eightyfour_old, color='k', linestyle='dashed')
plt.axvline(eightyfour_new, color='k', linestyle='dashed')
plt.legend()
plt.title(r'$\Sigma \chi^{2}$')
plt.show()
'''
gs = fig.axes[9].get_gridspec()
#gs = fig.axes.get_gridspec()


# remove the underlying axes
for ax in fig.axes[10:12]:
#for ax in fig.axes[1:, -1]:
    ax.remove()
axbig = fig.add_subplot(gs[10:12])
#axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5), xycoords='axes fraction', va='center')

#plt.sca(axbig)

plt.hist(chi_2_new,label='Oversampled Data Generation',color='green',histtype='step',density=True)
plt.hist(chi_2_old,label='Interpolated-only data',color='blue',histtype='step',density=True)
plt.axvline(sixteen_old, color='blue', linestyle='dashed')
plt.axvline(sixteen_new, color='green', linestyle='dashed')
plt.axvline(eightyfour_old, color='blue', linestyle='dashed')
plt.axvline(eightyfour_new, color='green', linestyle='dashed')
#plt.xlabel()
#plt.legend()
plt.xlabel(r'$\sum\limits^{f_{\mathrm{max}}}_{i=f_{\mathrm{min}}} \chi^{2}_{i}$',fontsize=12)
plt.xticks(fontsize=10)
#plt.yticks(fontsize=10)
plt.savefig('data_generation_combined.png')
plt.show()



'''
plt.hist(likelihood_new,label='over-sampled data generation')
plt.hist(likelihood_old,label='interpolation-only data generation')
plt.legend()
plt.title(r'$\log{\mathcal{L}}$')
plt.show()


#90% credible interval
fig = corner.corner(data_plot,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.05, 0.95],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='green')
corner.corner(data_plot_31,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.05, 0.95],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='blue',fig=fig)
plt.legend(handles=[blue_line,green_line], bbox_to_anchor=(0., 1.0, 1., .0), loc=4)
plt.savefig(dir+'corner_plot_compare_data_generation_90.png')
plt.show()
'''

'''
corner.corner(data_plot_true,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.16, 0.84],show_titles=False, title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='purple',fig=fig)
'''


'''
plt.hist(L_3_here,bins=20,density=True,label='X Channel')
plt.hist(L_3_here_31,bins=20,density=True,label='XYZ Channels',alpha=0.5)
plt.xlabel(r"$L_{3}-L_{3_{True}}$")
#plt.title('X Channel Only')
plt.legend()
plt.show()

plt.hist(L_2_here,bins=20,density=True,label='X Channel')
plt.hist(L_2_here_31,bins=20,density=True,label='XYZ Channels',alpha=0.5)
plt.xlabel(r"$L_{2}-L_{2_{True}}$")
#plt.title('X Channel Only')
plt.legend()
plt.show()
'''



'''
data_estimation = np.array([L_3,L_2,L_3_p,L_2_p])
print('likelihood')
print(likelihood)
#data_plot = np.vstack([likelihood,L_3,L_2,L_3_p,L_2_p])
print('data_plot')
print(data_plot)
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