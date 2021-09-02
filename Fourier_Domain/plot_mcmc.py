import sys
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import corner 




#data = np.recfromtxt('chain_data_overlap_add.dat',names = True)
data = np.recfromtxt('chainfile_testing_fourier_domain_runtime.dat',names = True)
#named this because it's the test described on page 191 in notebook
#dir = '/Users/jessica/Desktop/Project_1/Rigid_Rotation/plots/new_simulation/'
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
#dir = '/Users/jessica/Desktop/Project_1/Rigid_Rotation/split_interferometry/Production/Filter_Length_Tests/Plot/test_noise_matrix_equal_arm_967_25/'
dir='./Fourier_Domain_Results/'
likelihood = data['likelihood']
print('likelihood')
print(likelihood)
print('data')
print(data)
removed = int(0.25*len(likelihood))
likelihood = data['likelihood'][removed::]
print('likelihood')
print(likelihood)
L_3 = data['L_3'][removed::]
L_2 = data['L_2'][removed::]
L_3_p = data['L_3_p'][removed::]
L_2_p = data['L_2_p'][removed::]
L_1 = data['L_1'][removed::]
L_1_p = data['L_1_p'][removed::]
ar = data['Current_AR'][removed::]

chi_2 = data['sum_chi_2'][removed::]
log_term_determ = data['log_term_determ'][removed::]

L_1_real = 8.339095192
L_1_p_real = 8.338994879
L_2_real = 8.338867041
L_2_p_real = 8.339095192
L_3_real = 8.338994879
L_3_p_real = 8.338867041

#true_diff = L_1_real-L_2_real
maximum = likelihood.max()

print('max likelihood')
print(maximum)
#indice = np.where(likelihood>=-1)
indice = np.where(likelihood==maximum)[0]
print('indice')
print(indice)
print('L_3 vals at max L')
print(L_3[indice])
print('L_3 real')
print(L_3_real)
print('L_3 -L_3 true MAX L nanoseconds')
print((L_3[indice]-L_3_real)*1e9)
print('L_3 -L_3 true MEDIAN nanoseconds')
print((np.median(L_3)-L_3_real)*1e9)
print('L_2 vals at max L')
print(L_2[indice])
print('L_2 real')
print(L_2_real)
print('L_2 -L_2 true max L nanoseconds')
print((L_2[indice]-L_2_real)*1e9)
print('L_2 -L_2 true MEDIAN nanoseconds')
print((np.median(L_2)-L_2_real)*1e9)
print('L_1 vals at max L')
print(L_1[indice])
print('L_1 real')
print(L_1_real)
print('L_1 -L_1 true max L nanoseconds')
print((L_1[indice]-L_1_real)*1e9)
print('L_1 -L_1 true MEDIAN nanoseconds')
print((np.median(L_1)-L_1_real)*1e9)
print('L_3_p vals at max L')
print(L_3_p[indice])
print('L_3_p real')
print(L_3_p_real)
print('L_3_p -L_3_p true max L nanoseconds')
print((L_3_p[indice]-L_3_p_real)*1e9)
print('L_3_p -L_3_p true MEDIAN nanoseconds')
print((np.median(L_3_p)-L_3_p_real)*1e9)
print('L_2_p vals at max L')
print(L_2_p[indice])
print('L_2_p real')
print(L_2_p_real)
print('L_2_p -L_2_p true max L nanoseconds')
print((L_2_p[indice]-L_2_p_real)*1e9)
print('L_2_p -L_2_p true MEDIAN nanoseconds')
print((np.median(L_2_p)-L_2_p_real)*1e9)
print('L_1_p vals at max L')
print(L_1_p[indice])
print('L_1_p real')
print(L_1_p_real)
print('L_1_p -L_1_p true max L nanoseconds')
print((L_1_p[indice]-L_1_p_real)*1e9)
print('L_1_p -L_1_p true MEDIAN nanoseconds')
print((np.median(L_1_p)-L_1_p_real)*1e9)

#difference = L_1-L_2


index = np.argmax(likelihood)


plt.plot(likelihood)
plt.xlabel('iteration #')
plt.ylabel(r'$\log{\mathcal{L}}$')
plt.savefig(dir+'likelihood.png')
plt.show()

plt.plot(chi_2)
plt.xlabel('iteration #')
plt.ylabel(r'sum $\chi^{2}$')
plt.savefig(dir+'chi_2.png')
plt.show()

plt.plot(log_term_determ)
plt.xlabel('iteration #')
plt.ylabel(r'sum $\log{|\Sigma|}$')
plt.savefig(dir+'log_term_determinant.png')
plt.show()

plt.plot(ar)
plt.xlabel('iteration #')
plt.ylabel('AR @ iteration #')
plt.savefig(dir+'AR.png')
plt.show()

plt.plot(L_1-L_1_real)
#plt.title(r'$\sigma = 0.3 \mu s$')
plt.ylabel(r'$L_{1}-L_{1_{True}}$')
plt.savefig(dir+'L_1_diff.png')
plt.show()

plt.plot(L_1_p-L_1_p_real)
#plt.title(r'$\sigma = 0.3 \mu s$')
plt.ylabel(r"$L^{'}_{1}-L^{1}_{1_{True}}$")
plt.savefig(dir+'L_1_p_diff.png')
plt.show()

plt.plot(L_3-L_3_real)
#plt.title(r'$\sigma = 0.3 \mu s$')
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
bin = 20
plt.hist(likelihood,bins=bin)
plt.title('logL')
#plt.axvline(np.median(diff_L_2_L_2_p),label = 'median={0}'.format(np.median(diff_L_2_L_2_p)))
plt.legend()
#plt.xlabel(r"$L_{2}-L^{'}_{2}$")
plt.savefig(dir+'logL_hist.png')
plt.show()

plt.hist(chi_2,bins=bin)
#plt.plot(chi_2)
plt.title(r'sum $\chi^{2}$')
plt.savefig(dir+'chi_2_hist.png')
plt.show()

plt.hist(L_3-L_3_real,bins=bin)
plt.title(r"$L_{3}$")
#plt.axvline(0,label = r'$L_{3} = {0}$'.format(np.median(L_3_real))
#plt.legend()
plt.xlabel(r"$L_{3}-L_{3_{True}}$")
plt.savefig(dir+'L_3_hist.png')
plt.show()

plt.hist(L_2-L_2_real,bins=bin)
plt.title(r"$L_{2}$")
#plt.axvline(0,label = r'$L_{2} = {0}$'.format(np.median(L_2_real)))
#plt.legend()
plt.xlabel(r"$L_{2}-L_{2_{True}}$")
plt.savefig(dir+'L_2_hist.png')
plt.show()

plt.hist(L_3_p-L_3_p_real,bins=bin)
plt.title(r"$L^{'}_{3}$")
#plt.axvline(0,label = r"$L^{'}_{3} = {0}$".format(np.median(L_3_p_real)))
#plt.legend()
plt.xlabel(r"$L^{'}_{3}-L^{'}_{3_{True}}$")
plt.savefig(dir+'L_3_p_hist.png')
plt.show()

plt.hist(L_2_p-L_2_p_real,bins=bin)
plt.title(r"$L^{'}_{2}$")
#plt.axvline(0,label = r"$L^{'}_{2} = {2}$".format(np.median(L_2_p_real)))
#plt.legend()
plt.xlabel(r"$L^{'}_{2}-L^{'}_{2_{True}}$")
plt.savefig(dir+'L_2_p_hist.png')
plt.show()

plt.hist(L_1_p-L_1_p_real,bins=bin)
plt.title(r"$L^{'}_{1}$")
#plt.axvline(0,label = r"$L^{'}_{2} = {2}$".format(np.median(L_2_p_real)))
#plt.legend()
plt.xlabel(r"$L^{'}_{1}-L^{'}_{1_{True}}$")
plt.savefig(dir+'L_1_p_hist.png')
plt.show()

plt.hist(L_1-L_1_real,bins=bin)
plt.title(r"$L_{1}$")
#plt.axvline(0,label = r'$L_{2} = {0}$'.format(np.median(L_2_real)))
#plt.legend()
plt.xlabel(r"$L_{1}-L_{1_{True}}$")
plt.savefig(dir+'L_1_hist.png')
plt.show()
'''






'''
cm = plt.cm.get_cmap('RdYlBu')
#sc = plt.scatter(L_1[removed:-1:1], L_2[removed:-1:1], c=likelihood[removed:-1:1], s=5, cmap=cm,label='likelihood')
#sc = plt.scatter(L_3[indice]-L_3_real, L_2[indice]-L_2_real, c=likelihood[indice], s=5, cmap=cm,label='likelihood')
sc = plt.scatter(L_3-L_3_real, L_2-L_2_real, c=likelihood, s=5, cmap=cm,label='likelihood')
#plt.plot(L_1,L_2)
plt.colorbar(sc)
plt.xlabel(r'$L_{3}-L_{3_{True}}$')
plt.ylabel(r'$L_{2}-L_{2_{True}}$')
#plt.xlim(-2e-5,2.5e-5)
#plt.ylim(-2e-5,2e-5)
#plt.ticklabel_format(axis='both', style='sci',scilimits=(-6,6))
plt.savefig(dir+'Two_D_Likelihood_L_3_L_2.png')
plt.show()
'''

'''
cm = plt.cm.get_cmap('RdYlBu')
#sc = plt.scatter(L_1[removed:-1:1], L_2[removed:-1:1], c=likelihood[removed:-1:1], s=5, cmap=cm,label='likelihood')
#sc = plt.scatter(L_3[indice]-L_3_real, L_2[indice]-L_3[indice], c=likelihood[indice], s=5, cmap=cm,label='likelihood')
sc = plt.scatter(L_3-L_3_real, L_2-L_3, c=likelihood, s=5, cmap=cm,label='likelihood')

#plt.plot(L_1,L_2)
plt.colorbar(sc)
plt.xlabel(r'$L_{3}-L_{3_{True}}$')
plt.ylabel(r"$L_{2}-L_{3}$")

#plt.xlim(-2e-5,2.5e-5)
#plt.ylim(-2e-5,2e-5)

#plt.ticklabel_format(axis='both', style='sci',scilimits=(-6,6))
plt.savefig(dir+'Two_D_Likelihood_difference_L2_L3.png')
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